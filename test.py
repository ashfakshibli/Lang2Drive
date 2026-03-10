#!/usr/bin/env python3
"""Minimal OpenAI Responses API smoke test for image inputs."""

from __future__ import annotations

import argparse

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple image-question smoke test against the OpenAI Responses API.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument(
        "--image-url",
        default="https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg",
        help="Public image URL to send to the model",
    )
    parser.add_argument("--question", default="What teams are playing in this image?", help="Prompt text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": args.question},
                    {"type": "input_image", "image_url": args.image_url},
                ],
            }
        ],
    )
    print(getattr(response, "output_text", "") or "")


if __name__ == "__main__":
    main()
