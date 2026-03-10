#!/usr/bin/env python3
"""CLIP-based keyframe selection for Lang2Drive scenario image folders."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import open_clip
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


@dataclass
class KeyframeConfig:
    input_root: Path
    output_root: Path
    frames_subdir: Optional[str]
    top_k: int
    mmr_lambda: float
    novelty_weight: float
    model_name: str
    pretrained: str
    batch_size: int
    device: str
    use_fp16: bool
    copy_files: bool
    scenario_prompts_json: Optional[Path]


def normalize_scene_name(scene_name: str) -> str:
    return re.sub(r"[_-]+", " ", scene_name).strip()


def list_frames(frame_dir: Path) -> List[Path]:
    frames: List[Path] = []
    for pattern in IMAGE_PATTERNS:
        frames.extend(frame_dir.glob(pattern))
    return sorted(set(frames), key=numeric_sort_key)


def numeric_sort_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.name)
    if numbers:
        return int(numbers[-1]), path.name
    return 10**18, path.name


def load_prompt_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object mapping scenario names to prompts")
    prompt_map: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            prompt_map[key] = value.strip()
    return prompt_map


def prompt_ensemble(scene_name: str, scene_prompt: str) -> List[str]:
    base = re.sub(r"\s+", " ", (scene_prompt or normalize_scene_name(scene_name))).strip(" .")
    variants = [
        f"Dashcam view: {base}.",
        f"Driving scene with {base}.",
        f"Hazard ahead in the ego lane: {base}.",
    ]
    seen = set()
    deduped: List[str] = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


class CLIPKeyframeSelector:
    def __init__(self, config: KeyframeConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(config.model_name)
        self.model = self.model.to(self.device).eval()
        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        text_features = self.model.encode_text(tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path]) -> torch.Tensor:
        features: List[torch.Tensor] = []
        for start in range(0, len(image_paths), self.config.batch_size):
            batch_paths = image_paths[start : start + self.config.batch_size]
            images = []
            for image_path in batch_paths:
                image = Image.open(image_path).convert("RGB")
                images.append(self.preprocess(image))
            batch = torch.stack(images, dim=0).to(self.device)
            if self.config.use_fp16 and self.device.type == "cuda":
                batch = batch.half()
            image_features = self.model.encode_image(batch)
            features.append(image_features / image_features.norm(dim=-1, keepdim=True))
        return torch.cat(features, dim=0)

    @staticmethod
    def novelty_scores(image_features: torch.Tensor) -> torch.Tensor:
        if image_features.size(0) <= 1:
            return torch.zeros((image_features.size(0),), device=image_features.device, dtype=image_features.dtype)
        similarities = (image_features[1:] * image_features[:-1]).sum(dim=-1)
        novelty = torch.cat(
            [torch.zeros((1,), device=image_features.device, dtype=image_features.dtype), 1.0 - similarities],
            dim=0,
        )
        return novelty.clamp(0.0, 1.0)

    @staticmethod
    def select_mmr(image_features: torch.Tensor, scores: torch.Tensor, top_k: int, lam: float) -> List[int]:
        total = int(scores.size(0))
        if total == 0:
            return []
        top_k = min(top_k, total)
        first = int(torch.argmax(scores).item())
        selected = [first]
        remaining = torch.ones(total, dtype=torch.bool, device=image_features.device)
        remaining[first] = False

        while len(selected) < top_k:
            candidate_indices = torch.where(remaining)[0]
            if candidate_indices.numel() == 0:
                break
            selected_indices = torch.tensor(selected, device=image_features.device, dtype=torch.long)
            max_similarity = (image_features[candidate_indices] @ image_features[selected_indices].T).max(dim=1).values
            candidate_scores = scores[candidate_indices]
            mmr_scores = lam * candidate_scores - (1.0 - lam) * max_similarity
            best_candidate = candidate_indices[torch.argmax(mmr_scores).item()].item()
            selected.append(int(best_candidate))
            remaining[best_candidate] = False

        return selected

    def select_for_scenario(self, scene_name: str, scene_prompt: str, frame_paths: Sequence[Path]) -> List[tuple[Path, float]]:
        image_features = self.encode_images(frame_paths)
        text_features = self.encode_texts(prompt_ensemble(scene_name, scene_prompt))
        clip_scores = (image_features @ text_features.T).max(dim=1).values
        if self.config.novelty_weight > 0.0:
            novelty = self.novelty_scores(image_features)
            weight = self.config.novelty_weight
            final_scores = (1.0 - weight) * clip_scores + weight * novelty
        else:
            final_scores = clip_scores
        selected_indices = self.select_mmr(
            image_features=image_features,
            scores=final_scores,
            top_k=self.config.top_k,
            lam=self.config.mmr_lambda,
        )
        selected = [(frame_paths[index], float(final_scores[index].float().cpu().item())) for index in selected_indices]
        selected.sort(key=lambda item: item[1], reverse=True)
        return selected


def iter_scenario_dirs(root: Path) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry


def write_selection(
    config: KeyframeConfig,
    scene_name: str,
    selections: Sequence[tuple[Path, float]],
) -> None:
    scene_output_dir = config.output_root / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    rows = ["rank\tscore\toutput_filename\tsource_path"]
    for rank, (source_path, score) in enumerate(selections, start=1):
        destination_name = f"keyframe_{rank:02d}_score_{score:.5f}{source_path.suffix.lower()}"
        destination_path = scene_output_dir / destination_name
        source_rel = source_path.relative_to(config.input_root).as_posix()
        if config.copy_files:
            shutil.copy2(source_path, destination_path)
        rows.append(f"{rank}\t{score:.6f}\t{destination_name}\t{source_rel}")

    (scene_output_dir / "selected_keyframes.tsv").write_text("\n".join(rows), encoding="utf-8")


def parse_args() -> KeyframeConfig:
    parser = argparse.ArgumentParser(description="Select CLIP-ranked keyframes for scenario image folders.")
    parser.add_argument("--input_root", required=True, help="Root directory containing scenario folders")
    parser.add_argument("--output_root", required=True, help="Directory where selected keyframes are written")
    parser.add_argument(
        "--frames_subdir",
        default="rgb",
        help="Subdirectory inside each scenario folder that contains frames; use '' to read scenario roots directly",
    )
    parser.add_argument(
        "--scenario_prompts_json",
        default=None,
        help="Optional JSON mapping scenario directory names to natural-language prompts",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of keyframes to keep per scenario")
    parser.add_argument("--mmr_lambda", type=float, default=0.72, help="MMR relevance weight")
    parser.add_argument("--novelty_weight", type=float, default=0.25, help="Blend weight for frame novelty")
    parser.add_argument("--model_name", default="ViT-L-14", help="open_clip model name")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained weights identifier")
    parser.add_argument("--batch_size", type=int, default=64, help="Image encode batch size")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda",
    )
    parser.add_argument("--disable_fp16", action="store_true", help="Disable fp16 inference on CUDA")
    parser.add_argument("--tsv_only", action="store_true", help="Write scores only; do not copy image files")
    args = parser.parse_args()

    return KeyframeConfig(
        input_root=Path(args.input_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        frames_subdir=args.frames_subdir or None,
        top_k=max(1, args.top_k),
        mmr_lambda=max(0.0, min(1.0, args.mmr_lambda)),
        novelty_weight=max(0.0, min(1.0, args.novelty_weight)),
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=max(1, args.batch_size),
        device=args.device,
        use_fp16=not args.disable_fp16,
        copy_files=not args.tsv_only,
        scenario_prompts_json=Path(args.scenario_prompts_json).resolve() if args.scenario_prompts_json else None,
    )


def main() -> None:
    config = parse_args()
    if not config.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {config.input_root}")
    config.output_root.mkdir(parents=True, exist_ok=True)

    prompt_map = load_prompt_map(config.scenario_prompts_json)
    selector = CLIPKeyframeSelector(config)

    for scenario_dir in tqdm(list(iter_scenario_dirs(config.input_root)), desc="Scenarios"):
        frame_dir = scenario_dir / config.frames_subdir if config.frames_subdir else scenario_dir
        if not frame_dir.is_dir():
            print(f"[WARN] Skipping {scenario_dir.name}: missing frame directory {frame_dir}")
            continue

        frame_paths = list_frames(frame_dir)
        if not frame_paths:
            print(f"[WARN] Skipping {scenario_dir.name}: no frames found")
            continue

        scene_prompt = prompt_map.get(scenario_dir.name, normalize_scene_name(scenario_dir.name))
        selections = selector.select_for_scenario(scenario_dir.name, scene_prompt, frame_paths)
        write_selection(config, scenario_dir.name, selections)

    print(f"Saved selections to {config.output_root}")


if __name__ == "__main__":
    main()
