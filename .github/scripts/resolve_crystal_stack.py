"""Read an approved SGLang image without executing candidate configuration."""

import argparse
import json
import re
from pathlib import Path

IMAGE = re.compile(r"lmsysorg/sglang:v([0-9]+\.[0-9]+\.[0-9]+)-rocm[0-9.]+-mi35x@sha256:[0-9a-f]{64}")


def resolve(trusted: dict, candidate: dict) -> dict:
    image = trusted.get("sglang_image", "")
    match = IMAGE.fullmatch(image) if isinstance(image, str) else None
    if not match:
        raise ValueError("The reviewed stack must pin an approved SGLang ROCm image by digest.")
    if candidate.get("sglang_image") != image:
        raise ValueError("Candidate SGLang stack differs from the reviewed qualification stack.")
    return {"image": image, "version": match[1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(
            json.dumps(resolve(json.loads(args.trusted.read_text()), json.loads(args.candidate.read_text())))
        )
    except (OSError, ValueError, TypeError, AttributeError):
        parser.exit(1, "Qualification requires matching, approved, digest-pinned SGLang images.\n")
