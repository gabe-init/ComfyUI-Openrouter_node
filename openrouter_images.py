"""Dedicated OpenRouter image requests; no automatic retries of paid calls."""

import base64
import binascii
import math
from urllib.parse import urlparse

import requests

try:
    from . import openrouter_catalog as catalog
except ImportError:
    import openrouter_catalog as catalog


def _accepts(descriptor, value):
    if not isinstance(descriptor, dict):
        return False
    kind = descriptor.get("type")
    if kind == "enum":
        return value in descriptor.get("values", [])
    if kind == "range":
        return isinstance(value, int) and not isinstance(value, bool) and descriptor.get("min", value) <= value <= descriptor.get("max", value)
    # Boolean descriptors mean parameter presence is supported, not boolean input.
    return kind == "boolean"


def _endpoint_supports(endpoint, controls, reference_count):
    parameters = endpoint["supported_parameters"]
    if any(not _accepts(parameters.get(key), value) for key, value in controls.items()):
        return False
    references = parameters.get("input_references")
    if reference_count or references is not None:
        if not _accepts(references, reference_count):
            return False
    formats = parameters.get("output_format")
    if formats is not None:
        allowed = {"png", "webp"} if controls.get("background") == "transparent" else {"png", "jpeg", "webp"}
        if not any(_accepts(formats, value) for value in allowed):
            return False
    return True


def _reference_url(value):
    if not isinstance(value, str):
        raise ValueError("Image references must be image data URLs or HTTP(S) URLs.")
    try:
        parsed = urlparse(value)
        # Accessing port also validates malformed or out-of-range port values.
        parsed.port
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password and not any(char.isspace() for char in value):
            return value
    except ValueError:
        pass
    if value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
        try:
            if _is_raster(base64.b64decode(value.split(",", 1)[1], validate=True)):
                return value
        except (binascii.Error, ValueError):
            pass
    raise ValueError("Image references must be valid PNG/JPEG/WebP data URLs or HTTP(S) URLs.")


def _is_raster(data):
    return data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff") or (
        data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    )


def _select_format_and_routing(compatible, endpoints, transparent):
    candidates = []
    for preferred in (("png", "webp") if transparent else ("png", "webp", "jpeg")):
        matching = [item for item in compatible if _accepts(item["supported_parameters"].get("output_format"), preferred)]
        candidates.append((preferred, matching))
    # Without a descriptor the endpoint's default is the only valid request.
    candidates.append((None, [item for item in compatible if item["supported_parameters"].get("output_format") is None]))
    for output_format, matching in candidates:
        if not matching:
            continue
        if len(matching) == len(endpoints):
            return matching, output_format, None
        # A tag can cover multiple endpoints: every endpoint behind it must work.
        tags = {item["provider_tag"] for item in matching if isinstance(item.get("provider_tag"), str) and item["provider_tag"]}
        tags.difference_update(item.get("provider_tag") for item in endpoints if item not in matching)
        if tags:
            matching = [item for item in matching if item.get("provider_tag") in tags]
            return matching, output_format, {"only": sorted(tags)}
    raise ValueError("The compatible image provider cannot be selected safely; choose different image settings.")


def generate_images(api_key, model, prompt, reference_urls=None, aspect_ratio="auto", resolution="auto", quality="auto", background="auto", seed=0, timeout=120):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("An OpenRouter API key is required.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("An image prompt is required.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 3600:
        raise ValueError("Image request timeout must be between 1 and 3600 seconds.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("Image seed must be a nonnegative 64-bit integer.")
    if reference_urls is not None and not isinstance(reference_urls, (list, tuple)):
        raise ValueError("Image references must be a list of URLs.")
    references = [_reference_url(value) for value in (reference_urls or [])]
    if len(references) > 16:
        raise ValueError("OpenRouter accepts at most 16 image references; model limits may be lower.")
    catalog.require_model("image", model)
    endpoints = catalog.image_endpoints(model, timeout=timeout)
    controls = {}
    for key, value in (("aspect_ratio", aspect_ratio), ("resolution", resolution), ("quality", quality), ("background", background)):
        if value not in (None, "", "auto"):
            if not isinstance(value, str):
                raise ValueError(f"Image {key} must be a supported string value.")
            controls[key] = value
    if seed:
        controls["seed"] = seed
    compatible = [endpoint for endpoint in endpoints if _endpoint_supports(endpoint, controls, len(references))]
    if not compatible:
        settings = ", ".join(f"{key}={value}" for key, value in controls.items()) or "default settings"
        raise ValueError(f"No raster image endpoint for '{model}' supports {settings} with {len(references)} reference images. Check this model's supported settings and reference limits.")
    compatible, output_format, routing = _select_format_and_routing(compatible, endpoints, background == "transparent")
    payload = {"model": model, "prompt": prompt, **controls}
    if not seed and all(_accepts(item["supported_parameters"].get("seed"), seed) for item in compatible):
        payload["seed"] = seed
    if references:
        payload["input_references"] = [{"type": "image_url", "image_url": {"url": value}} for value in references]
    # Restrict routing when only some providers accept the requested combination.
    if routing:
        payload["provider"] = routing
    if output_format:
        payload["output_format"] = output_format
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/gabe-init/ComfyUI-Openrouter_node", "X-OpenRouter-Title": "ComfyUI OpenRouter Node"}
    try:
        response = requests.post(f"{catalog.API_BASE}/images", headers=headers, json=payload, timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        raise RuntimeError(f"Image request failed ({type(exc).__name__}); no automatic retry was made. Check OpenRouter activity before retrying.") from None
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"OpenRouter image request returned HTTP {response.status_code}. Check model settings and OpenRouter activity before retrying.")
    try:
        result = response.json()
    except ValueError:
        raise RuntimeError("OpenRouter returned invalid image JSON. Check OpenRouter activity before retrying.") from None
    if not isinstance(result, dict) or not isinstance(result.get("data"), list) or not result["data"]:
        raise RuntimeError("OpenRouter returned no generated images. Check OpenRouter activity before retrying.")
    images = []
    for item in result["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("b64_json"), str):
            raise RuntimeError("OpenRouter returned an invalid image entry.")
        if item.get("media_type") == "image/svg+xml":
            raise RuntimeError("Vector image output cannot be used as a ComfyUI raster image; choose a raster model.")
        try:
            decoded = base64.b64decode(item["b64_json"], validate=True)
        except (ValueError, binascii.Error):
            raise RuntimeError("OpenRouter returned invalid encoded image data.") from None
        if not _is_raster(decoded):
            raise RuntimeError("OpenRouter returned an unsupported image format; PNG, JPEG, and WebP are supported.")
        images.append(decoded)
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost):
        cost = None
    return {"images": images, "text": "", "usage": usage, "cost": cost}
