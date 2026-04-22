import base64
import hashlib
import io
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image

from .openrouter_catalog import OpenRouterCatalog

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    class VideoFromFile:  # pragma: no cover - local fallback for non-ComfyUI environments
        def __init__(self, file_obj):
            self.file_obj = file_obj


class OpenRouterVideoNode:
    """
    OpenRouter video generation node.
    """

    VIDEO_MODES = [
        "text_to_video",
        "image_to_video",
        "start_end_frame_to_video",
        "reference_to_video",
    ]

    API_BASE = "https://openrouter.ai/api/v1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (
                    OpenRouterCatalog.fetch_video_model_ids(),
                    {"video_capabilities": OpenRouterCatalog.fetch_video_widget_capabilities()},
                ),
                "mode": (cls.VIDEO_MODES, {"default": "text_to_video"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (OpenRouterCatalog.fetch_video_resolution_options(), {"default": "auto"}),
                "aspect_ratio": (OpenRouterCatalog.fetch_video_aspect_ratio_options(), {"default": "auto"}),
                "duration": (OpenRouterCatalog.fetch_video_duration_options(), {"default": "auto"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "poll_interval_seconds": ("INT", {"default": 30, "min": 5, "max": 300, "step": 1}),
                "timeout_seconds": ("INT", {"default": 900, "min": 30, "max": 7200, "step": 1}),
                "provider_json": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "Status", "Metadata", "Warnings")

    FUNCTION = "generate_video"
    CATEGORY = "LLM/Video"

    @classmethod
    def IS_CHANGED(
        cls,
        api_key,
        model,
        mode,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        generate_audio,
        seed,
        poll_interval_seconds,
        timeout_seconds,
        provider_json,
        image_1=None,
        image_2=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
    ):
        image_hashes = []
        for image in [
            image_1,
            image_2,
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
        ]:
            if isinstance(image, torch.Tensor):
                hasher = hashlib.sha256()
                hasher.update(image.cpu().numpy().tobytes())
                image_hashes.append(hasher.hexdigest())
            else:
                image_hashes.append(None)

        return (
            api_key,
            model,
            mode,
            prompt,
            resolution,
            aspect_ratio,
            duration,
            generate_audio,
            seed,
            poll_interval_seconds,
            timeout_seconds,
            provider_json,
            tuple(image_hashes),
        )

    @staticmethod
    def _headers(api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gabe-init/ComfyUI-Openrouter_node",
            "X-Title": "ComfyUI OpenRouter Video Node",
        }

    @staticmethod
    def _image_to_data_url(image: torch.Tensor) -> str:
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected a ComfyUI IMAGE tensor.")

        if image.ndim == 4:
            if image.shape[0] != 1:
                image = image[:1]
            image = image.squeeze(0)

        if image.ndim != 3:
            raise ValueError(f"Unexpected image dimensions: {image.shape}. Expected HWC.")

        image_np = image.cpu().numpy()
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)

        pil_image = Image.fromarray(image_np, "RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _provider_from_json(provider_json: str) -> Optional[Dict[str, Any]]:
        provider_json = provider_json.strip()
        if not provider_json:
            return None

        provider = json.loads(provider_json)
        if provider in ({}, []):
            return None
        if not isinstance(provider, dict):
            raise ValueError("provider_json must decode to a JSON object.")

        # Convenience: if the user pasted only the provider-options map,
        # wrap it into the shape expected by OpenRouter's videos API.
        known_provider_fields = {
            "order",
            "allow_fallbacks",
            "require_parameters",
            "data_collection",
            "zdr",
            "only",
            "ignore",
            "sort",
            "options",
        }
        if not any(key in provider for key in known_provider_fields):
            return {"options": provider}
        return provider

    @staticmethod
    def _connected_reference_images(kwargs: Dict[str, Any]) -> List[str]:
        connected = []
        for name in [
            "reference_image_1",
            "reference_image_2",
            "reference_image_3",
            "reference_image_4",
        ]:
            if kwargs.get(name) is not None:
                connected.append(name)
        return connected

    @staticmethod
    def _description_supports_reference_mode(description: str) -> bool:
        return OpenRouterCatalog._video_description_supports_reference_mode(description)

    @classmethod
    def _infer_supported_modes(cls, model_data: Dict[str, Any]) -> List[str]:
        return OpenRouterCatalog.infer_video_supported_modes(model_data)

    @staticmethod
    def _normalize_duration_input(duration: Any) -> Optional[int]:
        if duration in (None, "", "auto", 0, "0"):
            return None
        return int(duration)

    @staticmethod
    def _estimated_cost_display_text(estimated_cost: Optional[Dict[str, Any]]) -> Optional[str]:
        if not estimated_cost:
            return None
        display_text = estimated_cost.get("display_text")
        if isinstance(display_text, str) and display_text:
            return display_text
        return None

    @staticmethod
    def _actual_cost_display_text(poll_data: Dict[str, Any]) -> Optional[str]:
        usage = poll_data.get("usage")
        if isinstance(usage, dict):
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                return f"${cost:.4f}"
        if isinstance(usage, (int, float)):
            return f"${usage:.4f}"
        return None

    @classmethod
    def _build_payload(
        cls,
        model: str,
        mode: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: Any,
        generate_audio: bool,
        seed: int,
        provider_json: str,
        warnings: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        model_data = OpenRouterCatalog.get_video_model_by_id(model)
        supported_modes = cls._infer_supported_modes(model_data)
        supported_frame_images = set(model_data.get("supported_frame_images") or [])
        supported_resolutions = model_data.get("supported_resolutions") or []
        supported_aspect_ratios = model_data.get("supported_aspect_ratios") or []
        supported_durations = model_data.get("supported_durations") or []

        if not prompt.strip():
            raise ValueError("prompt is required for OpenRouter video generation.")

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "generate_audio": generate_audio,
        }

        requested_duration = cls._normalize_duration_input(duration)
        if requested_duration is None:
            if supported_durations:
                payload["duration"] = supported_durations[0]
                warnings.append(
                    f"Duration auto-selected to {supported_durations[0]} for {model}."
                )
            else:
                warnings.append(
                    "Duration left unspecified because the public catalog did not advertise supported durations."
                )
        else:
            payload["duration"] = requested_duration

        if resolution != "auto":
            if supported_resolutions and resolution not in supported_resolutions:
                raise ValueError(
                    f"Resolution '{resolution}' is not supported by {model}. "
                    f"Published values: {supported_resolutions}"
                )
            payload["resolution"] = resolution

        if aspect_ratio != "auto":
            if supported_aspect_ratios and aspect_ratio not in supported_aspect_ratios:
                raise ValueError(
                    f"Aspect ratio '{aspect_ratio}' is not supported by {model}. "
                    f"Published values: {supported_aspect_ratios}"
                )
            payload["aspect_ratio"] = aspect_ratio

        effective_duration = payload.get("duration")
        if (
            effective_duration is not None
            and supported_durations
            and effective_duration not in supported_durations
        ):
            raise ValueError(
                f"Duration '{effective_duration}' is not supported by {model}. "
                f"Published values: {supported_durations}"
            )

        if model_data.get("generate_audio") is False and generate_audio:
            warnings.append(f"{model} does not publicly advertise generated audio support.")

        if model_data.get("seed") is True:
            payload["seed"] = seed
        elif seed != 0:
            warnings.append(f"{model} does not publicly advertise seed support. Seed was omitted.")

        if mode == "image_to_video":
            if kwargs.get("image_1") is None:
                raise ValueError("Mode 'image_to_video' requires image_1.")
            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(kwargs["image_1"])},
                    "frame_type": "first_frame",
                }
            ]
            if "first_frame" not in supported_frame_images:
                warnings.append(
                    f"{model} does not publicly advertise 'first_frame' support."
                )

        elif mode == "start_end_frame_to_video":
            if kwargs.get("image_1") is None:
                raise ValueError("Mode 'start_end_frame_to_video' requires image_1 as the first frame.")
            if kwargs.get("image_2") is None:
                raise ValueError("Mode 'start_end_frame_to_video' requires image_2 as the last frame.")

            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(kwargs["image_1"])},
                    "frame_type": "first_frame",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(kwargs["image_2"])},
                    "frame_type": "last_frame",
                },
            ]

            if "first_frame" not in supported_frame_images:
                warnings.append(
                    f"{model} does not publicly advertise 'first_frame' support."
                )
            if "last_frame" not in supported_frame_images:
                warnings.append(
                    f"{model} does not publicly advertise 'last_frame' support."
                )

        elif mode == "reference_to_video":
            reference_names = cls._connected_reference_images(kwargs)
            if not reference_names:
                raise ValueError("Mode 'reference_to_video' requires at least one connected reference image.")

            payload["input_references"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(kwargs[name])},
                }
                for name in reference_names
            ]

            if "reference_to_video" not in supported_modes:
                warnings.append(
                    f"Reference-to-video is not explicitly confirmed by the public catalog for {model}."
                )

        if mode not in supported_modes:
            warnings.append(
                f"Selected mode '{mode}' is outside the model's publicly inferred supported modes: {supported_modes}."
            )

        provider = cls._provider_from_json(provider_json)
        if provider is not None:
            payload["provider"] = provider

        return payload

    @classmethod
    def _response_detail_text(cls, response: requests.Response) -> str:
        try:
            detail = response.json()
            return json.dumps(detail, ensure_ascii=True)
        except Exception:
            return response.text[:4000]

    @classmethod
    def _raise_submit_error(cls, response: requests.Response, payload: Dict[str, Any]) -> None:
        detail_text = cls._response_detail_text(response)
        payload_preview = json.dumps(payload, ensure_ascii=True)[:6000]
        raise RuntimeError(
            f"OpenRouter video submit failed ({response.status_code}). "
            f"Response: {detail_text} | Payload: {payload_preview}"
        )

    @classmethod
    def _submit_job(cls, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{cls.API_BASE}/videos",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if not response.ok:
            cls._raise_submit_error(response, payload)
        return response.json()

    @classmethod
    def _poll_job(
        cls,
        headers: Dict[str, str],
        job_id: str,
        polling_url: Optional[str],
        poll_interval_seconds: int,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_seconds
        url = polling_url or f"{cls.API_BASE}/videos/{job_id}"

        while True:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out while waiting for OpenRouter video job {job_id}.")

            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")

            if status == "completed":
                return data
            if status == "failed":
                raise RuntimeError(f"OpenRouter video job failed: {data.get('error', 'Unknown error')}")
            if status not in {"pending", "in_progress"}:
                raise RuntimeError(f"Unexpected OpenRouter video job status: {status}")

            time.sleep(poll_interval_seconds)

    @classmethod
    def _poll_job_once(
        cls,
        headers: Dict[str, str],
        job_id: str,
        polling_url: Optional[str],
    ) -> Dict[str, Any]:
        url = polling_url or f"{cls.API_BASE}/videos/{job_id}"
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()

    @classmethod
    def _download_url_needs_auth(cls, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        return parsed.netloc == "openrouter.ai"

    @classmethod
    def _extract_download_urls(cls, job_id: str, poll_data: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        seen = set()
        blocked_urls = {
            poll_data.get("polling_url"),
            f"{cls.API_BASE}/videos/{job_id}",
        }

        def add_url(value: Any) -> None:
            if not isinstance(value, str):
                return
            if not value.startswith(("http://", "https://")):
                return
            if value in blocked_urls or value in seen:
                return
            seen.add(value)
            urls.append(value)

        for key in ["unsigned_urls", "signed_urls", "urls"]:
            for value in poll_data.get(key) or []:
                add_url(value)

        for key in ["content_url", "download_url", "url"]:
            add_url(poll_data.get(key))

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, str) and "url" in nested_key.lower():
                        add_url(nested_value)
                    else:
                        walk(nested_value)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(poll_data.get("output"))
        walk(poll_data.get("outputs"))
        walk(poll_data.get("artifacts"))
        walk(poll_data.get("data"))

        add_url(f"{cls.API_BASE}/videos/{job_id}/content?index=0")
        return urls

    @classmethod
    def _download_from_url(
        cls,
        headers: Dict[str, str],
        url: str,
    ) -> requests.Response:
        request_headers = headers if cls._download_url_needs_auth(url) else None
        return requests.get(url, headers=request_headers, timeout=300)

    @classmethod
    def _download_video_bytes(
        cls,
        headers: Dict[str, str],
        job_id: str,
        polling_url: Optional[str],
        poll_data: Dict[str, Any],
        poll_interval_seconds: int,
    ) -> bytes:
        download_retry_seconds = max(3, min(10, poll_interval_seconds // 3 or 3))
        deadline = time.time() + 90
        last_poll_data = poll_data
        attempts: List[Dict[str, Any]] = []

        while True:
            download_urls = cls._extract_download_urls(job_id, last_poll_data)
            for url in download_urls:
                response = cls._download_from_url(headers, url)
                if response.ok:
                    return response.content

                attempts.append(
                    {
                        "url": url,
                        "status_code": response.status_code,
                        "detail": cls._response_detail_text(response)[:600],
                    }
                )

            if time.time() > deadline:
                attempt_preview = json.dumps(attempts[-5:], ensure_ascii=True)
                poll_preview = json.dumps(last_poll_data, ensure_ascii=True)[:4000]
                raise RuntimeError(
                    "OpenRouter video download failed after retries. "
                    f"job_id={job_id} | attempts={attempt_preview} | last_poll={poll_preview}"
                )

            time.sleep(download_retry_seconds)
            last_poll_data = cls._poll_job_once(headers, job_id, polling_url)
            if last_poll_data.get("status") == "failed":
                raise RuntimeError(
                    f"OpenRouter video job failed after completion while downloading content: "
                    f"{last_poll_data.get('error', 'Unknown error')}"
                )

    def generate_video(
        self,
        api_key,
        model,
        mode,
        prompt,
        resolution,
        aspect_ratio,
        duration,
        generate_audio,
        seed,
        poll_interval_seconds,
        timeout_seconds,
        provider_json,
        image_1=None,
        image_2=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
    ):
        if not api_key.strip():
            raise ValueError("api_key is required.")

        warnings: List[str] = []
        payload = self._build_payload(
            model=model,
            mode=mode,
            prompt=prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=duration,
            generate_audio=generate_audio,
            seed=seed,
            provider_json=provider_json,
            warnings=warnings,
            image_1=image_1,
            image_2=image_2,
            reference_image_1=reference_image_1,
            reference_image_2=reference_image_2,
            reference_image_3=reference_image_3,
            reference_image_4=reference_image_4,
        )
        model_data = OpenRouterCatalog.get_video_model_by_id(model)
        estimated_cost = OpenRouterCatalog.estimate_video_cost(model_data, payload, mode)

        headers = self._headers(api_key)
        submit_data = self._submit_job(headers, payload)
        job_id = submit_data.get("id")
        polling_url = submit_data.get("polling_url")
        status = submit_data.get("status", "pending")

        if not job_id:
            raise RuntimeError(f"OpenRouter video submit response is missing a job id: {submit_data}")

        poll_data = self._poll_job(
            headers=headers,
            job_id=job_id,
            polling_url=polling_url,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )

        video_bytes = self._download_video_bytes(
            headers=headers,
            job_id=job_id,
            polling_url=polling_url,
            poll_data=poll_data,
            poll_interval_seconds=poll_interval_seconds,
        )
        video_output = VideoFromFile(io.BytesIO(video_bytes))

        metadata = {
            "submit": submit_data,
            "final": poll_data,
            "estimated_cost": estimated_cost,
            "payload_summary": {
                "model": payload.get("model"),
                "mode": mode,
                "duration": payload.get("duration"),
                "resolution": payload.get("resolution"),
                "aspect_ratio": payload.get("aspect_ratio"),
                "generate_audio": payload.get("generate_audio"),
                "has_frame_images": bool(payload.get("frame_images")),
                "has_input_references": bool(payload.get("input_references")),
                "supported_sizes": model_data.get("supported_sizes"),
                "allowed_passthrough_parameters": model_data.get("allowed_passthrough_parameters"),
                "supports_background_control": model_data.get("supports_background_control"),
            },
        }

        estimated_cost_text = self._estimated_cost_display_text(estimated_cost)
        actual_cost_text = self._actual_cost_display_text(poll_data)
        status_text = (
            f"Completed | job_id={job_id} | model={model} | status={poll_data.get('status', status)}"
        )
        if estimated_cost_text:
            status_text += f" | est_cost={estimated_cost_text}"
        if actual_cost_text:
            status_text += f" | actual_cost={actual_cost_text}"
        metadata_text = json.dumps(metadata, indent=2, ensure_ascii=True)
        if estimated_cost_text:
            warnings.append(f"Estimated cost before submit: {estimated_cost_text}")
            if estimated_cost and estimated_cost.get("assumption"):
                warnings.append(str(estimated_cost["assumption"]))
        if actual_cost_text:
            warnings.append(f"OpenRouter reported actual cost: {actual_cost_text}")
        warnings_text = "\n".join(f"- {item}" for item in warnings) if warnings else "No warnings."
        return (video_output, status_text, metadata_text, warnings_text)


NODE_CLASS_MAPPINGS = {
    "OpenRouterVideoNode": OpenRouterVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterVideoNode": "OpenRouter Video",
}
