import math
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class OpenRouterCatalog:
    MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=all"
    VIDEO_MODELS_URL = "https://openrouter.ai/api/v1/videos/models"
    CACHE_DURATION = 3600
    VIDEO_RESOLUTION_ORDER = ["480p", "720p", "1080p", "1K", "2K", "4K"]
    VIDEO_ASPECT_RATIO_ORDER = ["1:1", "3:4", "4:3", "9:16", "16:9", "9:21", "21:9"]

    _all_models_cache: Optional[List[Dict[str, Any]]] = None
    _all_models_timestamp = 0.0
    _video_models_cache: Optional[List[Dict[str, Any]]] = None
    _video_models_timestamp = 0.0

    FALLBACK_CHAT_MODELS = [
        "anthropic/claude-sonnet-4.5",
        "black-forest-labs/flux.2-pro",
        "black-forest-labs/flux.2-flex",
        "google/gemini-2.5-pro",
        "openai/gpt-4o",
        "google/gemini-2.5-flash-image-preview",
    ]

    FALLBACK_VIDEO_MODELS = [
        "alibaba/wan-2.7",
        "alibaba/wan-2.6",
        "bytedance/seedance-2.0",
        "bytedance/seedance-2.0-fast",
        "bytedance/seedance-1-5-pro",
        "google/veo-3.1",
        "kwaivgi/kling-video-o1",
        "openai/sora-2-pro",
    ]

    @classmethod
    def _is_cache_valid(cls, timestamp: float) -> bool:
        return (time.time() - timestamp) <= cls.CACHE_DURATION

    @staticmethod
    def _fetch_json(url: str) -> Dict[str, Any]:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize_modalities(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        return []

    @staticmethod
    def _ordered_video_durations(values: Any) -> List[int]:
        durations = set()
        for value in values or []:
            try:
                durations.add(int(value))
            except (TypeError, ValueError):
                continue
        return sorted(durations)

    @staticmethod
    def _ordered_video_strings(values: Any) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for value in values or []:
            text = str(value)
            if text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    @classmethod
    def _ordered_video_resolutions(cls, values: Any) -> List[str]:
        resolutions = set(cls._ordered_video_strings(values))
        ordered = [value for value in cls.VIDEO_RESOLUTION_ORDER if value in resolutions]
        extras = sorted(resolutions.difference(ordered))
        return [*ordered, *extras]

    @classmethod
    def _ordered_video_aspect_ratios(cls, values: Any) -> List[str]:
        aspect_ratios = set(cls._ordered_video_strings(values))
        ordered = [value for value in cls.VIDEO_ASPECT_RATIO_ORDER if value in aspect_ratios]
        extras = sorted(aspect_ratios.difference(ordered))
        return [*ordered, *extras]

    @classmethod
    def _extract_output_modalities(cls, model: Dict[str, Any]) -> List[str]:
        architecture = model.get("architecture") or {}
        modalities = architecture.get("output_modalities")
        if modalities is None:
            modalities = model.get("output_modalities")
        return cls._normalize_modalities(modalities)

    @classmethod
    def fetch_all_models(cls) -> List[Dict[str, Any]]:
        if cls._all_models_cache is not None and cls._is_cache_valid(cls._all_models_timestamp):
            return cls._all_models_cache

        try:
            result = cls._fetch_json(cls.MODELS_URL)
            models = result.get("data", [])
            if isinstance(models, list) and models:
                cls._all_models_cache = models
                cls._all_models_timestamp = time.time()
                return models
        except requests.exceptions.RequestException as exc:
            print(f"Error fetching OpenRouter models catalog: {exc}")

        if cls._all_models_cache is not None:
            return cls._all_models_cache

        return []

    @classmethod
    def fetch_chat_model_ids(cls) -> List[str]:
        models = cls.fetch_all_models()
        if not models:
            return cls.FALLBACK_CHAT_MODELS

        model_ids = []
        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue

            output_modalities = set(cls._extract_output_modalities(model))

            # Keep the chat node focused on outputs it can already parse.
            if not output_modalities or output_modalities.intersection({"text", "image"}):
                model_ids.append(model_id)

        model_ids = sorted(set(model_ids))
        return model_ids if model_ids else cls.FALLBACK_CHAT_MODELS

    @classmethod
    def fetch_image_generation_model_ids(cls) -> List[str]:
        models = cls.fetch_all_models()
        if not models:
            return [
                model_id
                for model_id in cls.FALLBACK_CHAT_MODELS
                if "flux" in model_id or "image" in model_id
            ]

        model_ids = []
        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue

            output_modalities = set(cls._extract_output_modalities(model))
            if "image" in output_modalities:
                model_ids.append(model_id)

        model_ids = sorted(set(model_ids))
        return model_ids if model_ids else cls.FALLBACK_CHAT_MODELS

    @classmethod
    def fetch_chat_widget_capabilities(cls) -> Dict[str, Dict[str, Any]]:
        capabilities: Dict[str, Dict[str, Any]] = {}
        for model in cls.fetch_all_models():
            model_id = model.get("id")
            if not isinstance(model_id, str):
                continue

            output_modalities = cls._extract_output_modalities(model)
            output_modalities_set = set(output_modalities)
            capabilities[model_id] = {
                "output_modalities": output_modalities,
                "supports_image_generation": "image" in output_modalities_set,
                "supports_text_output": "text" in output_modalities_set or not output_modalities,
                "is_image_only": output_modalities_set == {"image"},
            }

        for model_id in cls.FALLBACK_CHAT_MODELS:
            capabilities.setdefault(
                model_id,
                {
                    "output_modalities": [],
                    "supports_image_generation": ("flux" in model_id or "image" in model_id),
                    "supports_text_output": "flux" not in model_id,
                    "is_image_only": "flux" in model_id,
                },
            )

        return capabilities

    @classmethod
    def get_chat_model_by_id(cls, model_id: str) -> Dict[str, Any]:
        for model in cls.fetch_all_models():
            if model.get("id") == model_id:
                return model

        return {
            "id": model_id,
            "name": model_id,
            "output_modalities": [],
            "architecture": {"output_modalities": []},
        }

    @classmethod
    def fetch_video_models(cls) -> List[Dict[str, Any]]:
        if cls._video_models_cache is not None and cls._is_cache_valid(cls._video_models_timestamp):
            return cls._video_models_cache

        try:
            result = cls._fetch_json(cls.VIDEO_MODELS_URL)
            models = result.get("data", [])
            if isinstance(models, list) and models:
                cls._video_models_cache = models
                cls._video_models_timestamp = time.time()
                return models
        except requests.exceptions.RequestException as exc:
            print(f"Error fetching OpenRouter video models catalog: {exc}")

        if cls._video_models_cache is not None:
            return cls._video_models_cache

        return []

    @classmethod
    def fetch_video_model_ids(cls) -> List[str]:
        video_models = cls.fetch_video_models()
        if not video_models:
            return cls.FALLBACK_VIDEO_MODELS

        model_ids = sorted(
            {
                model.get("id")
                for model in video_models
                if isinstance(model, dict) and isinstance(model.get("id"), str)
            }
        )
        return model_ids if model_ids else cls.FALLBACK_VIDEO_MODELS

    @staticmethod
    def _video_description_supports_reference_mode(description: str) -> bool:
        lowered = (description or "").lower()
        return (
            "reference-to-video" in lowered
            or "reference images" in lowered
            or "multimodal reference" in lowered
        )

    @staticmethod
    def _video_supports_background_control(model: Dict[str, Any]) -> bool:
        description = (model.get("description") or "").lower()
        allowed_passthrough_parameters = {
            value.lower()
            for value in OpenRouterCatalog._ordered_video_strings(
                model.get("allowed_passthrough_parameters")
            )
        }
        background_parameters = {
            "background",
            "background_mode",
            "remove_background",
            "transparent_background",
            "alpha_background",
        }
        background_phrases = [
            "transparent background",
            "alpha background",
            "remove background",
            "background removal",
        ]
        return bool(
            allowed_passthrough_parameters.intersection(background_parameters)
            or any(phrase in description for phrase in background_phrases)
        )

    @staticmethod
    def _parse_price(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_video_size(size: str) -> Optional[Tuple[int, int]]:
        if not isinstance(size, str) or "x" not in size:
            return None
        width_text, height_text = size.lower().split("x", 1)
        try:
            return int(width_text), int(height_text)
        except ValueError:
            return None

    @staticmethod
    def _simplify_ratio(width: int, height: int) -> str:
        divisor = math.gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    @classmethod
    def _size_matches_resolution(cls, size: str, resolution: Optional[str]) -> bool:
        if resolution in (None, "", "auto"):
            return True

        parsed = cls._parse_video_size(size)
        if parsed is None:
            return False

        width, height = parsed
        short_side = min(width, height)
        long_side = max(width, height)
        normalized = resolution.lower()

        if normalized == "480p":
            return short_side == 480
        if normalized == "720p":
            return short_side == 720
        if normalized == "1080p":
            return short_side == 1080
        if normalized == "4k":
            return long_side in {3840, 4096} or short_side in {2160}
        return True

    @classmethod
    def _size_matches_aspect_ratio(cls, size: str, aspect_ratio: Optional[str]) -> bool:
        if aspect_ratio in (None, "", "auto"):
            return True

        parsed = cls._parse_video_size(size)
        if parsed is None:
            return False

        width, height = parsed
        return cls._simplify_ratio(width, height) == aspect_ratio

    @classmethod
    def _matching_supported_sizes(
        cls,
        model: Dict[str, Any],
        resolution: Optional[str],
        aspect_ratio: Optional[str],
    ) -> List[str]:
        matches = []
        for size in cls._ordered_video_strings(model.get("supported_sizes")):
            if not cls._size_matches_resolution(size, resolution):
                continue
            if not cls._size_matches_aspect_ratio(size, aspect_ratio):
                continue
            matches.append(size)
        return matches

    @staticmethod
    def _resolution_suffixes(resolution: Optional[str]) -> List[str]:
        if resolution in (None, "", "auto"):
            return []

        normalized = resolution.lower()
        mapping = {
            "480p": ["480p"],
            "720p": ["720p"],
            "1080p": ["1080p"],
            "1k": ["1k", "1024p"],
            "2k": ["2k", "2048p"],
            "4k": ["4k", "2160p"],
        }
        return mapping.get(normalized, [normalized])

    @classmethod
    def _estimate_token_priced_video_cost(
        cls,
        model: Dict[str, Any],
        pricing_skus: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        duration = payload.get("duration")
        if not isinstance(duration, int):
            return None

        generate_audio = bool(payload.get("generate_audio", True))
        sku_key = "video_tokens"
        if not generate_audio and "video_tokens_without_audio" in pricing_skus:
            sku_key = "video_tokens_without_audio"

        rate = cls._parse_price(pricing_skus.get(sku_key))
        if rate is None:
            return None

        resolution = payload.get("resolution")
        aspect_ratio = payload.get("aspect_ratio")
        matched_sizes = cls._matching_supported_sizes(model, resolution, aspect_ratio)
        if not matched_sizes:
            matched_sizes = cls._ordered_video_strings(model.get("supported_sizes"))
        if not matched_sizes:
            return None

        estimates = []
        for size in matched_sizes:
            parsed = cls._parse_video_size(size)
            if parsed is None:
                continue
            width, height = parsed
            token_count = (width * height * duration * 24) / 1024
            estimates.append(
                {
                    "size": size,
                    "tokens": int(token_count),
                    "cost_usd": token_count * rate,
                }
            )

        if not estimates:
            return None

        costs = [item["cost_usd"] for item in estimates]
        return {
            "pricing_mode": "video_tokens",
            "sku_key": sku_key,
            "min_cost_usd": min(costs),
            "max_cost_usd": max(costs),
            "matched_sizes": matched_sizes,
            "size_estimates": estimates,
            "assumption": (
                "Select both resolution and aspect ratio to tighten the estimate."
                if len(estimates) > 1
                else "Estimate based on the selected size constraints."
            ),
        }

    @classmethod
    def _estimate_mode_resolution_priced_video_cost(
        cls,
        model: Dict[str, Any],
        pricing_skus: Dict[str, Any],
        payload: Dict[str, Any],
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        duration = payload.get("duration")
        if not isinstance(duration, int):
            return None

        prefix = "text_to_video" if mode == "text_to_video" else "image_to_video"
        candidate_resolutions = (
            [payload.get("resolution")]
            if payload.get("resolution") not in (None, "", "auto")
            else cls._ordered_video_resolutions(model.get("supported_resolutions"))
        )

        matches = []
        for resolution in candidate_resolutions:
            for suffix in cls._resolution_suffixes(resolution):
                sku_key = f"{prefix}_duration_seconds_{suffix}"
                rate = cls._parse_price(pricing_skus.get(sku_key))
                if rate is None:
                    continue
                matches.append(
                    {
                        "resolution": resolution,
                        "sku_key": sku_key,
                        "cost_usd": duration * rate,
                    }
                )
                break

        if not matches:
            return None

        costs = [item["cost_usd"] for item in matches]
        return {
            "pricing_mode": "mode_resolution_duration_seconds",
            "min_cost_usd": min(costs),
            "max_cost_usd": max(costs),
            "resolution_estimates": matches,
            "assumption": (
                "Estimate spans the published resolution-dependent SKUs for this mode."
                if len(matches) > 1
                else "Estimate based on the selected mode and resolution."
            ),
        }

    @classmethod
    def _estimate_duration_priced_video_cost(
        cls,
        model: Dict[str, Any],
        pricing_skus: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        duration = payload.get("duration")
        if not isinstance(duration, int):
            return None

        generate_audio = bool(payload.get("generate_audio", True))
        candidate_resolutions = (
            [payload.get("resolution")]
            if payload.get("resolution") not in (None, "", "auto")
            else cls._ordered_video_resolutions(model.get("supported_resolutions")) or [None]
        )

        matches = []
        for resolution in candidate_resolutions:
            suffixes = cls._resolution_suffixes(resolution)
            key_candidates = []

            if generate_audio:
                key_candidates.extend(
                    [f"duration_seconds_with_audio_{suffix}" for suffix in suffixes]
                )
                key_candidates.append("duration_seconds_with_audio")
            else:
                key_candidates.extend(
                    [f"duration_seconds_without_audio_{suffix}" for suffix in suffixes]
                )
                key_candidates.append("duration_seconds_without_audio")

            key_candidates.extend([f"duration_seconds_{suffix}" for suffix in suffixes])
            key_candidates.append("duration_seconds")

            for sku_key in key_candidates:
                rate = cls._parse_price(pricing_skus.get(sku_key))
                if rate is None:
                    continue
                matches.append(
                    {
                        "resolution": resolution,
                        "sku_key": sku_key,
                        "cost_usd": duration * rate,
                    }
                )
                break

        if not matches:
            return None

        costs = [item["cost_usd"] for item in matches]
        return {
            "pricing_mode": "duration_seconds",
            "min_cost_usd": min(costs),
            "max_cost_usd": max(costs),
            "resolution_estimates": matches,
            "assumption": (
                "Estimate spans the published duration-based SKUs for the selected settings."
                if len(matches) > 1
                else "Estimate based on the selected duration settings."
            ),
        }

    @classmethod
    def estimate_video_cost(
        cls,
        model: Dict[str, Any],
        payload: Dict[str, Any],
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        pricing_skus = model.get("pricing_skus") or {}
        if not isinstance(pricing_skus, dict) or not pricing_skus:
            return None

        estimate = None
        if any(key.startswith(("text_to_video_duration_seconds_", "image_to_video_duration_seconds_")) for key in pricing_skus):
            estimate = cls._estimate_mode_resolution_priced_video_cost(model, pricing_skus, payload, mode)
        elif "video_tokens" in pricing_skus or "video_tokens_without_audio" in pricing_skus:
            estimate = cls._estimate_token_priced_video_cost(model, pricing_skus, payload)
        else:
            estimate = cls._estimate_duration_priced_video_cost(model, pricing_skus, payload)

        if estimate is None:
            return None

        min_cost = estimate.get("min_cost_usd")
        max_cost = estimate.get("max_cost_usd")
        if min_cost is None or max_cost is None:
            return None

        if abs(min_cost - max_cost) < 1e-9:
            estimate["display_text"] = f"${min_cost:.4f}"
        else:
            estimate["display_text"] = f"${min_cost:.4f}-${max_cost:.4f}"

        estimate["pricing_skus"] = pricing_skus
        return estimate

    @classmethod
    def infer_video_supported_modes(cls, model: Dict[str, Any]) -> List[str]:
        modes = ["text_to_video"]
        frame_images = set(model.get("supported_frame_images") or [])
        description = model.get("description", "")

        if "first_frame" in frame_images:
            modes.append("image_to_video")
        if {"first_frame", "last_frame"}.issubset(frame_images):
            modes.append("start_end_frame_to_video")
        if cls._video_description_supports_reference_mode(description):
            modes.append("reference_to_video")
        return modes

    @classmethod
    def fetch_video_duration_options(cls) -> List[str]:
        durations = set()
        for model in cls.fetch_video_models():
            durations.update(cls._ordered_video_durations(model.get("supported_durations")))
        ordered = [str(value) for value in sorted(durations)]
        return ["auto", *ordered] if ordered else ["auto"]

    @classmethod
    def fetch_video_resolution_options(cls) -> List[str]:
        resolutions = set()
        for model in cls.fetch_video_models():
            resolutions.update(cls._ordered_video_resolutions(model.get("supported_resolutions")))
        ordered = cls._ordered_video_resolutions(resolutions)
        return ["auto", *ordered] if ordered else ["auto"]

    @classmethod
    def fetch_video_aspect_ratio_options(cls) -> List[str]:
        aspect_ratios = set()
        for model in cls.fetch_video_models():
            aspect_ratios.update(cls._ordered_video_aspect_ratios(model.get("supported_aspect_ratios")))
        ordered = cls._ordered_video_aspect_ratios(aspect_ratios)
        return ["auto", *ordered] if ordered else ["auto"]

    @classmethod
    def fetch_video_widget_capabilities(cls) -> Dict[str, Dict[str, Any]]:
        capabilities: Dict[str, Dict[str, Any]] = {}
        for model in cls.fetch_video_models():
            model_id = model.get("id")
            if not isinstance(model_id, str):
                continue

            supported_durations = [
                str(value) for value in cls._ordered_video_durations(model.get("supported_durations"))
            ]
            supported_resolutions = cls._ordered_video_resolutions(model.get("supported_resolutions"))
            supported_aspect_ratios = cls._ordered_video_aspect_ratios(model.get("supported_aspect_ratios"))
            supported_sizes = cls._ordered_video_strings(model.get("supported_sizes"))
            allowed_passthrough_parameters = cls._ordered_video_strings(
                model.get("allowed_passthrough_parameters")
            )

            capabilities[model_id] = {
                "supported_modes": cls.infer_video_supported_modes(model),
                "supported_durations": supported_durations,
                "supported_resolutions": supported_resolutions,
                "supported_aspect_ratios": supported_aspect_ratios,
                "supported_sizes": supported_sizes,
                "minimum_duration": supported_durations[0] if supported_durations else None,
                "generate_audio": model.get("generate_audio"),
                "pricing_skus": model.get("pricing_skus") if isinstance(model.get("pricing_skus"), dict) else {},
                "allowed_passthrough_parameters": allowed_passthrough_parameters,
                "supports_background_control": cls._video_supports_background_control(model),
            }
        return capabilities

    @classmethod
    def get_video_model_by_id(cls, model_id: str) -> Dict[str, Any]:
        for model in cls.fetch_video_models():
            if model.get("id") == model_id:
                normalized = dict(model)
                normalized["supported_resolutions"] = cls._ordered_video_resolutions(
                    model.get("supported_resolutions")
                )
                normalized["supported_aspect_ratios"] = cls._ordered_video_aspect_ratios(
                    model.get("supported_aspect_ratios")
                )
                normalized["supported_durations"] = cls._ordered_video_durations(
                    model.get("supported_durations")
                )
                normalized["supported_sizes"] = cls._ordered_video_strings(
                    model.get("supported_sizes")
                )
                normalized["supported_frame_images"] = cls._ordered_video_strings(
                    model.get("supported_frame_images")
                )
                normalized["allowed_passthrough_parameters"] = cls._ordered_video_strings(
                    model.get("allowed_passthrough_parameters")
                )
                normalized["supports_background_control"] = cls._video_supports_background_control(
                    model
                )
                return normalized

        # Fallback object so the video node still works offline.
        return {
            "id": model_id,
            "name": model_id,
            "supported_resolutions": [],
            "supported_aspect_ratios": [],
            "supported_durations": [],
            "supported_sizes": [],
            "supported_frame_images": [],
            "generate_audio": None,
            "seed": None,
            "allowed_passthrough_parameters": [],
            "supports_background_control": False,
            "description": "",
        }
