#!/usr/bin/env python3
"""Run GPT-based preannotation over an image tree."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Prompts:
    system: str
    general_perception: str
    regional_perception: str
    actionable_suggestion: str


def coerce_prompt_text(value: Any, key: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if parts:
            return "\n".join(parts)
    raise ValueError(f"prompts.json key '{key}' must be a non-empty string or list of strings")


def load_prompts(path: Path) -> Prompts:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Prompts(
        system=coerce_prompt_text(data.get("system"), "system"),
        general_perception=coerce_prompt_text(data.get("general_perception"), "general_perception"),
        regional_perception=coerce_prompt_text(data.get("regional_perception"), "regional_perception"),
        actionable_suggestion=coerce_prompt_text(data.get("actionable_suggestion"), "actionable_suggestion"),
    )


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def safe_json_loads(raw_text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(raw_text), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def call_with_retries(callable_fn: Callable[[], str], retries: int, base_sleep: float) -> str:
    for attempt in range(retries + 1):
        try:
            return callable_fn()
        except Exception:  # noqa: BLE001
            if attempt >= retries:
                raise
            sleep_for = base_sleep * (2 ** attempt) + random.uniform(0.0, 0.25)
            time.sleep(sleep_for)
    raise RuntimeError("unreachable")


def run_one_prompt(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    temperature: float,
    retries: int,
    base_sleep: float,
) -> dict[str, Any]:
    def do_request() -> str:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
            temperature=temperature,
            text={"format": {"type": "json_object"}},
        )
        return getattr(response, "output_text", "") or ""

    raw_text = call_with_retries(do_request, retries=retries, base_sleep=base_sleep)
    parsed, parse_error = safe_json_loads(raw_text)
    return {"raw_text": raw_text, "json": parsed, "json_error": parse_error}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPT-based preannotations for an image tree.")
    parser.add_argument("--input_dir", required=True, help="Root directory containing image files")
    parser.add_argument("--output_dir", required=True, help="Root directory for output JSON annotations")
    parser.add_argument("--prompts", required=True, help="Path to prompts.json")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSON files")
    parser.add_argument("--retries", type=int, default=6, help="Maximum number of retries per API call")
    parser.add_argument("--base_sleep", type=float, default=1.0, help="Base backoff sleep in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    prompts_path = Path(args.prompts).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts = load_prompts(prompts_path)
    client = OpenAI()
    images = list(iter_images(input_dir))
    if not images:
        print(f"No images found under {input_dir}")
        return

    for image_path in tqdm(images, desc="Annotating images"):
        relative_image = image_path.relative_to(input_dir)
        output_path = (output_dir / relative_image).with_suffix(".json")
        ensure_parent(output_path)
        if output_path.exists() and not args.overwrite:
            continue

        image_data_url = image_to_data_url(image_path)
        general = run_one_prompt(
            client=client,
            model=args.model,
            system_prompt=prompts.system,
            user_prompt=prompts.general_perception,
            image_data_url=image_data_url,
            temperature=args.temperature,
            retries=args.retries,
            base_sleep=args.base_sleep,
        )
        regional = run_one_prompt(
            client=client,
            model=args.model,
            system_prompt=prompts.system,
            user_prompt=prompts.regional_perception,
            image_data_url=image_data_url,
            temperature=args.temperature,
            retries=args.retries,
            base_sleep=args.base_sleep,
        )
        actionable = run_one_prompt(
            client=client,
            model=args.model,
            system_prompt=prompts.system,
            user_prompt=prompts.actionable_suggestion,
            image_data_url=image_data_url,
            temperature=args.temperature,
            retries=args.retries,
            base_sleep=args.base_sleep,
        )

        payload = {
            "meta": {
                "image_path": relative_image.as_posix(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "temperature": args.temperature,
                "prompts_file": prompts_path.name,
            },
            "general_perception": general["json"] if general["json"] is not None else general["raw_text"],
            "regional_perception": regional["json"] if regional["json"] is not None else regional["raw_text"],
            "actionable_suggestion": actionable["json"] if actionable["json"] is not None else actionable["raw_text"],
            "parse_debug": {
                "general_json_error": general["json_error"],
                "regional_json_error": regional["json_error"],
                "actionable_json_error": actionable["json_error"],
            },
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved annotations to {output_dir}")


if __name__ == "__main__":
    main()
