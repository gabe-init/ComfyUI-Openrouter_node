"""Public OpenRouter discovery with nonblocking UI snapshots.

Catalogs are kept separate: image capability descriptors are not interchangeable
with chat supported-parameter lists. Network work never runs in get_catalog().
"""

import copy
import threading
import time
from urllib.parse import quote

import requests


API_BASE = "https://openrouter.ai/api/v1"
CATALOG_URLS = {
    "chat": f"{API_BASE}/models",
    "image": f"{API_BASE}/images/models",
    "video": f"{API_BASE}/videos/models",
    "speech": f"{API_BASE}/models?output_modalities=speech",
}
CACHE_SECONDS = 15 * 60
FAILURE_BACKOFF_SECONDS = 30
DISCOVERY_TIMEOUT = 20
# These catalog entries require video/audio inputs absent from this node's modes.
UNSUPPORTED_VIDEO_MODELS = frozenset({
    "black-forest-labs/flux-video-edit", "black-forest-labs/flux-video-upscale",
    "runway/aleph-2", "heygen/avatar-iv",
})
_condition = threading.Condition()
_records = {kind: [] for kind in CATALOG_URLS}
_success_at = {kind: None for kind in CATALOG_URLS}
_attempt_at = {kind: None for kind in CATALOG_URLS}
_errors = {}
_refreshing = False
# A separate lock prevents endpoint discovery from blocking UI snapshots.
_endpoint_condition = threading.Condition()
_endpoints = {}
_endpoint_attempt_at = {}
_endpoint_errors = {}
_endpoint_inflight = set()


def _validate_kind(kind):
    if kind not in CATALOG_URLS:
        raise ValueError("Catalog kind must be chat, image, video, or speech.")


def _snapshot():
    return {
        **copy.deepcopy(_records),
        "errors": dict(_errors),
        "updated_at": max((stamp for stamp in _success_at.values() if stamp is not None), default=None),
    }


def _due(kind, now, force=False):
    last_attempt = _attempt_at[kind]
    if kind in _errors and last_attempt is not None and now - last_attempt < FAILURE_BACKOFF_SECONDS:
        return False
    last_success = _success_at[kind]
    return force or last_success is None or now - last_success >= CACHE_SECONDS


def _read_json(url, timeout=DISCOVERY_TIMEOUT):
    response = requests.get(url, timeout=timeout, allow_redirects=False)
    response.raise_for_status()
    if not 200 <= response.status_code < 300:
        raise ValueError("Unexpected discovery redirect.")
    return response.json()


def _error_text(exc):
    response = getattr(exc, "response", None)
    if response is not None:
        return f"OpenRouter discovery returned HTTP {response.status_code}."
    return f"OpenRouter discovery unavailable ({type(exc).__name__})."


def _perform_refresh(kinds):
    global _refreshing
    try:
        for kind in kinds:
            with _condition:
                _attempt_at[kind] = time.time()
            try:
                payload = _read_json(CATALOG_URLS[kind])
                models = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(models, list) or any(
                    not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"]
                    for model in models
                ):
                    raise ValueError("Invalid model catalog.")
                with _condition:
                    _records[kind] = copy.deepcopy(models)
                    _success_at[kind] = time.time()
                    _errors.pop(kind, None)
            except (requests.RequestException, ValueError, TypeError) as exc:
                with _condition:
                    _errors[kind] = _error_text(exc)
    finally:
        with _condition:
            _refreshing = False
            _condition.notify_all()


def get_catalog(refresh=False):
    """Return a snapshot immediately and schedule at most one refresh worker."""
    global _refreshing
    with _condition:
        kinds = [kind for kind in CATALOG_URLS if _due(kind, time.time(), refresh)]
        if kinds and not _refreshing:
            _refreshing = True
            worker = threading.Thread(target=_perform_refresh, args=(kinds,), daemon=True, name="openrouter-catalog")
            try:
                worker.start()
            except RuntimeError:
                _refreshing = False
                _condition.notify_all()
                raise
        return _snapshot()


def refresh_catalog(force=False):
    """Blocking refresh for execution/workers; never call on the HTTP event loop."""
    global _refreshing
    with _condition:
        if _refreshing:
            _condition.wait_for(lambda: not _refreshing)
            return _snapshot()
        kinds = [kind for kind in CATALOG_URLS if _due(kind, time.time(), force)]
        if not kinds:
            return _snapshot()
        _refreshing = True
    _perform_refresh(kinds)
    with _condition:
        return _snapshot()


def _find_model(snapshot, kind, model_id):
    for model in snapshot[kind]:
        if model["id"] == model_id:
            return model
    detail = snapshot["errors"].get(kind)
    suffix = f" {detail}" if detail else " Refresh models and choose an available model."
    raise ValueError(f"OpenRouter {kind} model '{model_id}' is not in the available catalog.{suffix}")


def get_model(kind, model_id):
    _validate_kind(kind)
    return _find_model(get_catalog(), kind, model_id)


def model_ids(kind):
    _validate_kind(kind)
    return sorted({model["id"] for model in get_catalog()[kind]})


def is_supported_video_model(record):
    """Exclude known edit/upscale/avatar operations from generation selectors."""
    return bool(
        isinstance(record, dict)
        and isinstance(record.get("id"), str)
        and record["id"]
        and record["id"] not in UNSUPPORTED_VIDEO_MODELS
        and not record.get("upscale_factor")
        and not record.get("creativity")
    )


def video_generation_models(snapshot=None):
    """Return supported generation records without altering the raw catalog."""
    if snapshot is None:
        snapshot = get_catalog()
    return [record for record in snapshot["video"] if is_supported_video_model(record)]


def require_model(kind, model_id):
    """Get discovery metadata, waiting for initial/stale discovery at execution."""
    _validate_kind(kind)
    return _find_model(refresh_catalog(), kind, model_id)


def image_endpoints(model_id, timeout=DISCOVERY_TIMEOUT):
    """Fetch definitive endpoint descriptors, without merging model-level unions."""
    if not isinstance(model_id, str) or len(model_id.split("/")) < 2 or any(
        part in {"", ".", ".."} for part in model_id.split("/")
    ):
        raise ValueError("Invalid OpenRouter image model ID.")
    with _endpoint_condition:
        _endpoint_condition.wait_for(lambda: model_id not in _endpoint_inflight)
        cached = _endpoints.get(model_id)
        if cached and time.time() - cached[0] < CACHE_SECONDS:
            return copy.deepcopy(cached[1])
        attempt = _endpoint_attempt_at.get(model_id)
        if model_id in _endpoint_errors and attempt is not None and time.time() - attempt < FAILURE_BACKOFF_SECONDS:
            raise ValueError(_endpoint_errors[model_id])
        _endpoint_inflight.add(model_id)
        _endpoint_attempt_at[model_id] = time.time()
    try:
        url = f"{API_BASE}/images/models/{quote(model_id, safe='/')}/endpoints"
        payload = _read_json(url, timeout=min(timeout, DISCOVERY_TIMEOUT))
        records = payload.get("endpoints") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records or any(
            not isinstance(item, dict) or not isinstance(item.get("supported_parameters"), dict)
            for item in records
        ):
            raise ValueError("Invalid image endpoint capabilities.")
        with _endpoint_condition:
            _endpoints[model_id] = (time.time(), copy.deepcopy(records))
            _endpoint_errors.pop(model_id, None)
        return copy.deepcopy(records)
    except (requests.RequestException, ValueError, TypeError) as exc:
        message = f"Cannot verify image settings for '{model_id}': {_error_text(exc)} Try refreshing models."
        with _endpoint_condition:
            _endpoint_errors[model_id] = message
        raise ValueError(message) from None
    finally:
        with _endpoint_condition:
            _endpoint_inflight.discard(model_id)
            _endpoint_condition.notify_all()
