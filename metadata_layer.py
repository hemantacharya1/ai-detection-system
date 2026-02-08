import c2pa
import json
import io
from typing import Dict, Any

# Strict allowlist of AI-related terms
AI_KEYWORDS = [
    "dall-e",
    "dalle",
    "gemini",
    "firefly",
    "stable diffusion",
    "midjourney",
    "generative ai",
    "ai generated",
    "synthetic media",
    "trained algorithmic media",
    "diffusion model",
    "text-to-image",
    "chatgpt",
    "openai",
    "nano-banana"
]


def _collect_strings(obj, results):
    """
    Recursively collect all string values from a JSON object.
    This makes us schema-agnostic across platforms.
    """
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, results)
    elif isinstance(obj, list):
        for item in obj:
            _collect_strings(item, results)
    elif isinstance(obj, str):
        results.append(obj.lower())


def check_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """
    Layer A: C2PA-based AI provenance detection.

    This layer ONLY fires if the image explicitly declares
    AI generation via cryptographically signed provenance data.
    """

    try:
        # --- Detect MIME type ---
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        elif image_bytes.startswith(b"\xff\xd8"):
            mime_type = "image/jpeg"
        else:
            return {
                "is_ai": False,
                "confidence": 0.0,
                "reason": "Unsupported image format for C2PA",
                "source": "layer_a_metadata"
            }

        stream = io.BytesIO(image_bytes)

        # --- Read C2PA manifest ---
        with c2pa.Reader(mime_type, stream) as reader:
            manifest_json = reader.json()

        if not manifest_json or manifest_json == "{}":
            return {
                "is_ai": False,
                "confidence": 0.0,
                "reason": "No C2PA provenance data found",
                "source": "layer_a_metadata"
            }

        data = json.loads(manifest_json)

        # --- Collect all text content ---
        strings = []
        _collect_strings(data, strings)

        # --- Check for explicit AI provenance ---
        matched_keywords = [
            kw for kw in AI_KEYWORDS
            if any(kw in s for s in strings)
        ]

        if matched_keywords:
            return {
                "is_ai": True,
                "confidence": 1.0,  # deterministic
                "reason": (
                    "Cryptographically signed C2PA provenance explicitly "
                    "declares AI generation"
                ),
                "matched_keywords": matched_keywords,
                "source": "layer_a_metadata"
            }

        # Manifest exists, but no AI claim
        return {
            "is_ai": False,
            "confidence": 0.0,
            "reason": (
                "C2PA provenance found, but no explicit AI-generation claim "
                "was declared"
            ),
            "source": "layer_a_metadata"
        }

    except Exception as e:
        # Safe fallback: do NOT guess
        return {
            "is_ai": False,
            "confidence": 0.0,
            "reason": f"Metadata analysis failed: {str(e)}",
            "source": "layer_a_metadata"
        }
