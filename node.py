import os
import requests
import json
import time
import base64
import io
import hashlib
import numpy as np
import torch
import tiktoken
from PIL import Image
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .chat_manager import ChatSessionManager
from .openrouter_catalog import OpenRouterCatalog

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    class VideoFromFile:
        def __init__(self, file_obj):
            self.file_obj = file_obj

PDF_DATA_TYPE = "*"

VIDEO_MODES = [
    "text_to_video",
    "image_to_video",
    "start_end_frame_to_video",
    "reference_to_video",
]


class OpenRouterNode:
    """
    Unified OpenRouter node supporting chat, image generation, and video generation.
    Use the request_type input to switch between "chat", "image", and "video" modes.
    """

    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.chat_manager = ChatSessionManager()

    @classmethod
    def _api_key_file_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), "openrouter_api_key.txt")

    @classmethod
    def _read_saved_api_key(cls) -> str:
        try:
            with open(cls._api_key_file_path(), "r", encoding="utf-8") as f:
                return f.read().strip()
        except (FileNotFoundError, OSError):
            return ""

    @staticmethod
    def _read_env_api_key() -> str:
        return os.environ.get("OPENROUTER_API_KEY", "").strip()

    @classmethod
    def _save_api_key(cls, api_key: str) -> None:
        try:
            with open(cls._api_key_file_path(), "w", encoding="utf-8") as f:
                f.write(api_key.strip())
        except OSError as e:
            print(f"Warning: Could not save API key to file: {e}")

    @classmethod
    def mask_api_key(cls, key: str) -> str:
        if not key or len(key) < 8:
            return ""
        return key[:4] + "..." + key[-4:]

    def _resolve_api_key(self, api_key: str) -> str:
        if api_key and api_key.strip() and "..." not in api_key:
            self._save_api_key(api_key.strip())
            return api_key.strip()
        saved = self._read_saved_api_key()
        if saved:
            return saved
        return self._read_env_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "request_type": (["chat", "image", "video"], {"default": "chat"}),
                "model": (
                    cls.fetch_unified_models(),
                    {
                        "chat_capabilities": OpenRouterCatalog.fetch_chat_widget_capabilities(),
                        "image_capabilities": OpenRouterCatalog.fetch_image_widget_capabilities(),
                        "video_capabilities": OpenRouterCatalog.fetch_video_widget_capabilities(),
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),

                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "user_message_box": ("STRING", {
                    "multiline": True,
                    "default": "Hello, how are you?"
                }),
                "image_generation_only": ("BOOLEAN", {"default": False}),
                "web_search": ("BOOLEAN", {"default": False}),
                "cheapest": ("BOOLEAN", {"default": True}),
                "fastest": ("BOOLEAN", {"default": False}),
                "aspect_ratio": ([
                    "auto",
                    "1:1 (1024x1024)",
                    "2:3 (832x1248)",
                    "3:2 (1248x832)",
                    "3:4 (864x1184)",
                    "4:3 (1184x864)",
                    "4:5 (896x1152)",
                    "5:4 (1152x896)",
                    "9:16 (768x1344)",
                    "16:9 (1344x768)",
                    "21:9 (1536x672)",
                    "1:4 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
                    "4:1 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
                    "1:8 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
                    "8:1 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
                ], {"default": "auto"}),
                "image_resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "round": 0.01,
                }),
                "pdf_engine": (["auto", "mistral-ocr", "pdf-text"], {"default": "auto"}),
                "chat_mode": ("BOOLEAN", {"default": False}),

                "video_mode": (VIDEO_MODES, {"default": "text_to_video"}),
                "video_prompt": ("STRING", {"multiline": True, "default": ""}),
                "video_resolution": (OpenRouterCatalog.fetch_video_resolution_options(), {"default": "auto"}),
                "video_aspect_ratio": (OpenRouterCatalog.fetch_video_aspect_ratio_options(), {"default": "auto"}),
                "duration": (OpenRouterCatalog.fetch_video_duration_options(), {"default": "auto"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "poll_interval_seconds": ("INT", {"default": 30, "min": 5, "max": 300, "step": 1}),
                "timeout_seconds": ("INT", {"default": 900, "min": 30, "max": 7200, "step": 1}),
                "provider_json": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "pdf_data": (PDF_DATA_TYPE,),
                "user_message_input": ("STRING", {"forceInput": True}),
                "video_frame_1": ("IMAGE",),
                "video_frame_2": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Output", "image", "video", "Stats", "Credits", "Warnings")
    FUNCTION = "generate_response"
    CATEGORY = "LLM"

    @classmethod
    def fetch_unified_models(cls):
        model_list = OpenRouterCatalog.fetch_unified_model_ids()
        return model_list if model_list else ["error_fetching_models"]

    @classmethod
    def fetch_openrouter_models(cls):
        return cls.fetch_unified_models()

    def validate_temperature(self, temperature):
        try:
            temp = float(temperature)
            return max(0.0, min(2.0, temp))
        except (ValueError, TypeError):
            return 1.0

    def fetch_credits(self, api_key):
        if not api_key:
            return "API Key not provided."

        url = "https://openrouter.ai/api/v1/credits"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/comfyui-openrouter",
            "X-Title": "ComfyUI OpenRouter LLM Node",
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            result = response.json()
            if "data" in result and "total_credits" in result["data"] and "total_usage" in result["data"]:
                total_credits = result["data"]["total_credits"]
                total_usage = result["data"]["total_usage"]
                remaining = total_credits - total_usage
                credits_text = f"Remaining: ${remaining:.3f}"
            else:
                credits_text = "Could not parse credit data from response."

            return credits_text

        except requests.exceptions.RequestException as e:
            error_message = f"Error fetching credits: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f" | Status Code: {e.response.status_code} | Response: {e.response.text[:200]}"
            return error_message
        except json.JSONDecodeError:
            return "Error fetching credits: Could not decode JSON response."

    @staticmethod
    def image_to_base64(image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input 'image' is not a torch.Tensor")

        if image.ndim == 4:
            if image.shape[0] != 1:
                print(f"Warning: Image batch size is {image.shape[0]}, using only the first image.")
            image = image.squeeze(0)

        if image.ndim != 3:
            raise ValueError(f"Unexpected image dimensions: {image.shape}. Expected HWC.")

        image_np = image.cpu().numpy()
        if image_np.dtype != np.uint8:
            if image_np.min() < 0 or image_np.max() > 1:
                print("Warning: Image tensor values outside [0, 1] range. Clamping.")
                image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)

        pil_image = Image.fromarray(image_np, 'RGB')

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")

        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def base64_to_image(base64_str: str) -> torch.Tensor:
        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            img = img.convert("RGB")

            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            print(f"Successfully converted base64 to image tensor: {img_tensor.shape}")
            return img_tensor

        except Exception as e:
            print(f"Error in base64_to_image: {e}")
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

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
    def count_tokens(text, model):
        if not text or not isinstance(text, str):
            return 0

        base_model = model.split(':')[0] if ':' in model else model

        encoding_name = "cl100k_base"
        try:
            cl100k_models = [
                "openai/gpt-4", "openai/gpt-3.5", "openai/gpt-4o",
                "anthropic/claude",
                "google/gemini",
                "meta-llama/llama-2", "meta-llama/llama-3",
                "mistralai/mistral", "mistralai/mixtral",
            ]
            is_cl100k = any(base_model.startswith(prefix) for prefix in cl100k_models)

            if is_cl100k:
                encoding_name = "cl100k_base"

            encoding = tiktoken.get_encoding(encoding_name)
            token_count = len(encoding.encode(text, disallowed_special=()))
            return token_count

        except Exception as e:
            print(f"Warning: Tiktoken error for model '{model}' (base: '{base_model}', encoding: '{encoding_name}'): {e}. Falling back to estimation.")
            return max(1, round(len(text) / 4))

    @staticmethod
    def is_image_generation_model(model_data):
        output_modalities = set(OpenRouterCatalog._extract_output_modalities(model_data))
        return "image" in output_modalities

    @staticmethod
    def looks_like_image_generation_request(text):
        if not text or not isinstance(text, str):
            return False

        lowered = text.lower()
        generation_markers = [
            "generate", "create", "draw", "make", "produce",
            "design", "render", "illustrate", "image of", "picture of",
            "photo of", "poster of", "logo of", "portrait of",
        ]
        return any(marker in lowered for marker in generation_markers)

    @staticmethod
    def normalize_aspect_ratio(aspect_ratio):
        if not aspect_ratio or aspect_ratio == "auto":
            return None
        if "(" in aspect_ratio:
            return aspect_ratio.split("(", 1)[0].strip()
        return aspect_ratio.strip()

    @staticmethod
    def _headers(api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gabe-init/ComfyUI-Openrouter_node",
            "X-Title": "ComfyUI OpenRouter Node",
        }

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

        known_provider_fields = {
            "order", "allow_fallbacks", "require_parameters",
            "data_collection", "zdr", "only", "ignore", "sort", "options",
        }
        if not any(key in provider for key in known_provider_fields):
            return {"options": provider}
        return provider

    @staticmethod
    def _connected_reference_images(kwargs: Dict[str, Any]) -> List[str]:
        connected = []
        for name in [
            "reference_image_1", "reference_image_2",
            "reference_image_3", "reference_image_4",
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

        image_1 = kwargs.get("image_1")
        image_2 = kwargs.get("image_2")

        if mode == "image_to_video":
            if image_1 is None:
                raise ValueError("Mode 'image_to_video' requires video_frame_1.")
            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(image_1)},
                    "frame_type": "first_frame",
                }
            ]
            if "first_frame" not in supported_frame_images:
                warnings.append(
                    f"{model} does not publicly advertise 'first_frame' support."
                )

        elif mode == "start_end_frame_to_video":
            if image_1 is None:
                raise ValueError("Mode 'start_end_frame_to_video' requires video_frame_1 as the first frame.")
            if image_2 is None:
                raise ValueError("Mode 'start_end_frame_to_video' requires video_frame_2 as the last frame.")

            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(image_1)},
                    "frame_type": "first_frame",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(image_2)},
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

    def generate_response(
        self,
        api_key,
        request_type,
        model,
        seed,
        system_prompt,
        user_message_box,
        image_generation_only,
        web_search,
        cheapest,
        fastest,
        aspect_ratio,
        image_resolution,
        temperature,
        pdf_engine,
        chat_mode,
        video_mode,
        video_prompt,
        video_resolution,
        video_aspect_ratio,
        duration,
        generate_audio,
        poll_interval_seconds,
        timeout_seconds,
        provider_json,
        pdf_data=None,
        user_message_input=None,
        video_frame_1=None,
        video_frame_2=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        **kwargs,
    ):
        api_key = self._resolve_api_key(api_key)
        if request_type == "video":
            return self._generate_video(
                api_key=api_key,
                model=model,
                seed=seed,
                video_mode=video_mode,
                video_prompt=video_prompt,
                video_resolution=video_resolution,
                video_aspect_ratio=video_aspect_ratio,
                duration=duration,
                generate_audio=generate_audio,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
                provider_json=provider_json,
                video_frame_1=video_frame_1,
                video_frame_2=video_frame_2,
                reference_image_1=reference_image_1,
                reference_image_2=reference_image_2,
                reference_image_3=reference_image_3,
                reference_image_4=reference_image_4,
            )
        elif request_type == "image":
            return self._generate_image(
                api_key=api_key,
                model=model,
                seed=seed,
                user_message_box=user_message_box,
                cheapest=cheapest,
                fastest=fastest,
                aspect_ratio=aspect_ratio,
                image_resolution=image_resolution,
                temperature=temperature,
                user_message_input=user_message_input,
                **kwargs,
            )
        else:
            return self._generate_chat(
                api_key=api_key,
                model=model,
                seed=seed,
                system_prompt=system_prompt,
                user_message_box=user_message_box,
                image_generation_only=image_generation_only,
                web_search=web_search,
                cheapest=cheapest,
                fastest=fastest,
                aspect_ratio=aspect_ratio,
                image_resolution=image_resolution,
                temperature=temperature,
                pdf_engine=pdf_engine,
                chat_mode=chat_mode,
                pdf_data=pdf_data,
                user_message_input=user_message_input,
                **kwargs,
            )

    def _generate_chat(
        self,
        api_key,
        model,
        seed,
        system_prompt,
        user_message_box,
        image_generation_only,
        web_search,
        cheapest,
        fastest,
        aspect_ratio,
        image_resolution,
        temperature,
        pdf_engine,
        chat_mode,
        pdf_data=None,
        user_message_input=None,
        **kwargs,
    ):
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        placeholder_video = None
        if not api_key:
            return ("Error: API Key not provided.", placeholder_image, placeholder_video, "Stats N/A", "Credits N/A", "")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/comfyui-openrouter",
            "X-Title": "ComfyUI OpenRouter LLM Node",
        }

        validated_temp = self.validate_temperature(temperature)

        user_text = user_message_input if user_message_input is not None and user_message_input.strip() else user_message_box

        session_path = None

        if chat_mode:
            session_path, messages = self.chat_manager.get_or_create_session(user_text, system_prompt)

            if messages and messages[0]["role"] == "system" and messages[0]["content"] != system_prompt:
                messages[0]["content"] = system_prompt
        else:
            messages = [
                {"role": "system", "content": system_prompt},
            ]

        user_content_blocks = []

        user_content_blocks.append({
            "type": "text",
            "text": user_text
        })

        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')],
                           key=lambda x: int(x.split('_')[1]))

        for image_key in image_keys:
            if kwargs[image_key] is not None:
                try:
                    img_str = self.image_to_base64(kwargs[image_key])
                    user_content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    })
                except Exception as e:
                    print(f"Error processing {image_key}: {e}")
                    return (f"Error processing {image_key}: {e}", placeholder_image, placeholder_video, "Stats N/A", "Credits N/A", "")

        pdf_filename = "document.pdf"
        if pdf_data is not None:
            if isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
                pdf_bytes = pdf_data["bytes"]
                if "filename" in pdf_data and isinstance(pdf_data["filename"], str) and pdf_data["filename"].strip():
                    pdf_filename = pdf_data["filename"]

                try:
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    data_url = f"data:application/pdf;base64,{base64_pdf}"
                    user_content_blocks.append({
                        "type": "file",
                        "file": {
                            "filename": pdf_filename,
                            "file_data": data_url
                        }
                    })
                except Exception as e:
                    print(f"Error encoding PDF: {e}")
                    return (f"Error encoding PDF: {e}", placeholder_image, placeholder_video, "Stats N/A", "Credits N/A", "")
            else:
                print(f"Warning: pdf_data input is not in the expected format (dict with 'filename' and 'bytes'). PDF not included.")

        has_multimodal_content = len(user_content_blocks) > 1 or any(block.get("type") != "text" for block in user_content_blocks)

        if has_multimodal_content:
            new_user_message = {
                "role": "user",
                "content": user_content_blocks
            }
        else:
            new_user_message = {
                "role": "user",
                "content": user_text
            }

        if chat_mode:
            messages.append(new_user_message)
        else:
            messages.append(new_user_message)

        modified_model = model
        if web_search and ":online" not in modified_model:
            modified_model = f"{modified_model}:online"
        if ":online" not in modified_model:
            if cheapest and ":floor" not in modified_model:
                modified_model = f"{modified_model}:floor"
            elif fastest and not cheapest and ":nitro" not in modified_model:
                modified_model = f"{modified_model}:nitro"

        selected_model = OpenRouterCatalog.get_chat_model_by_id(model)
        model_output_modalities = set(OpenRouterCatalog._extract_output_modalities(selected_model))

        data = {
            "model": modified_model,
            "messages": messages,
            "temperature": validated_temp,
            "seed": seed
        }

        image_generation_requested = (
            self.is_image_generation_model(selected_model) and (
                image_generation_only or self.looks_like_image_generation_request(user_text)
            )
        )

        if image_generation_requested:
            if model_output_modalities == {"image"}:
                data["modalities"] = ["image"]
            else:
                data["modalities"] = ["image", "text"]

            image_config = {}
            normalized_aspect_ratio = self.normalize_aspect_ratio(aspect_ratio)
            if normalized_aspect_ratio is not None:
                image_config["aspect_ratio"] = normalized_aspect_ratio
            if image_resolution:
                image_config["image_size"] = image_resolution
            if image_config:
                data["image_config"] = image_config

        print(f"Payload: model={modified_model}")

        if pdf_engine != "auto":
            data["plugins"] = [
                {
                    "id": "file-parser",
                    "pdf": {
                        "engine": pdf_engine
                    }
                }
            ]

        text_token_estimate = 0
        try:
            text_token_estimate = self.count_tokens(system_prompt, model) + self.count_tokens(user_text, model)
        except Exception as e:
            print(f"Warning: Token counting failed - {e}")

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            end_time = time.time()

            result = response.json()
            debug_str = json.dumps(result, default=str)
            print(f"API response ({len(debug_str)} chars): {debug_str[:500]}")

            if not result.get("choices") or not result["choices"][0].get("message"):
                raise ValueError("Invalid response format from API: 'choices' or 'message' missing.")

            message = result["choices"][0]["message"]
            text_output = message.get("content", "")
            image_tensor = placeholder_image

            if message.get("images"):
                print(f"Found {len(message['images'])} image(s) in API response")
                try:
                    first_image = message["images"][0]
                    image_url = first_image["image_url"]["url"]

                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",", 1)[1]
                        try:
                            image_tensor = self.base64_to_image(base64_str)
                            print(f"Successfully decoded image from API response")
                        except Exception as e:
                            print(f"Error decoding image: {e}")
                    else:
                        print(f"Image URL format not supported: {image_url[:50]}...")
                except Exception as e:
                    print(f"Error processing images from response: {e}")
            else:
                print("No images found in API response - this may be normal if the model doesn't support image generation or the prompt didn't request an image")

            if isinstance(text_output, list):
                text_parts = []
                for content in text_output:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            text_parts.append(content.get("text", ""))
                        elif content.get("type") == "image_url":
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                base64_str = image_url.split(",", 1)[1]
                                try:
                                    image_tensor = self.base64_to_image(base64_str)
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                text_output = "\n".join(text_parts)

            response_ms = result.get("response_ms", None)
            api_usage = result.get("usage", {})
            prompt_tokens = api_usage.get("prompt_tokens", text_token_estimate)
            completion_tokens = api_usage.get("completion_tokens", 0)
            if completion_tokens == 0 and text_output:
                try:
                    completion_tokens = self.count_tokens(text_output, model)
                except Exception as e:
                    print(f"Warning: Completion token counting failed - {e}")

            tps = 0
            elapsed_time = end_time - start_time
            if response_ms is not None:
                server_elapsed_time = response_ms / 1000.0
                if server_elapsed_time > 0:
                    tps = completion_tokens / server_elapsed_time
            elif elapsed_time > 0:
                tps = completion_tokens / elapsed_time

            stats_text = (
                f"TPS: {tps:.2f}, "
                f"Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, "
                f"Temp: {validated_temp:.1f}, "
                f"Model: {modified_model}"
            )
            if pdf_engine != "auto":
                stats_text += f", PDF Engine: {pdf_engine}"
            if image_generation_requested:
                stats_text += ", Image Generation: enabled"

            credits_text = self.fetch_credits(api_key)

            if chat_mode and session_path:
                assistant_message = {
                    "role": "assistant",
                    "content": text_output
                }
                messages.append(assistant_message)

                self.chat_manager.save_conversation(session_path, messages)

            return (text_output, image_tensor, placeholder_video, stats_text, credits_text, "")

        except requests.exceptions.RequestException as e:
            error_message = f"API Request Error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_message += f" | Details: {error_detail}"
                except json.JSONDecodeError:
                    error_message += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}"
            else:
                error_message += " (Network or connection issue)"
            print(f"ERROR: {error_message}")
            return (error_message, placeholder_image, placeholder_video, "Stats N/A due to error", "Credits N/A due to error", "")
        except Exception as e:
            print(f"ERROR: Node Error: {str(e)}")
            return (f"Node Error: {str(e)}", placeholder_image, placeholder_video, "Stats N/A due to error", "Credits N/A due to error", "")

    def _generate_image(
        self,
        api_key,
        model,
        seed,
        user_message_box,
        cheapest,
        fastest,
        aspect_ratio,
        image_resolution,
        temperature,
        user_message_input=None,
        **kwargs,
    ):
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        placeholder_video = None
        if not api_key:
            return ("Error: API Key not provided.", placeholder_image, placeholder_video, "Stats N/A", "Credits N/A", "")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/comfyui-openrouter",
            "X-Title": "ComfyUI OpenRouter Image Node",
        }

        validated_temp = self.validate_temperature(temperature)

        user_text = user_message_input if user_message_input is not None and user_message_input.strip() else user_message_box

        if not user_text.strip():
            return ("Error: A prompt is required for image generation.", placeholder_image, placeholder_video, "Stats N/A", "Credits N/A", "")

        messages = [
            {"role": "user", "content": user_text}
        ]

        user_content_blocks = [{"type": "text", "text": user_text}]

        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')],
                           key=lambda x: int(x.split('_')[1]))

        for image_key in image_keys:
            if kwargs[image_key] is not None:
                try:
                    img_str = self.image_to_base64(kwargs[image_key])
                    user_content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    })
                except Exception as e:
                    print(f"Error processing {image_key}: {e}")
                    return (f"Error processing {image_key}: {e}", placeholder_image, placeholder_video, "Stats N/A", "Credits N/A", "")

        has_images = len(user_content_blocks) > 1
        if has_images:
            messages = [{"role": "user", "content": user_content_blocks}]

        modified_model = model
        if ":online" not in modified_model:
            if cheapest and ":floor" not in modified_model:
                modified_model = f"{modified_model}:floor"
            elif fastest and not cheapest and ":nitro" not in modified_model:
                modified_model = f"{modified_model}:nitro"

        data = {
            "model": modified_model,
            "messages": messages,
            "temperature": validated_temp,
            "seed": seed,
            "modalities": ["image"],
        }

        image_config = {}
        normalized_aspect_ratio = self.normalize_aspect_ratio(aspect_ratio)
        if normalized_aspect_ratio is not None:
            image_config["aspect_ratio"] = normalized_aspect_ratio
        if image_resolution:
            image_config["image_size"] = image_resolution
        if image_config:
            data["image_config"] = image_config

        print(f"Payload: model={modified_model}, modalities=image")

        text_token_estimate = 0
        try:
            text_token_estimate = self.count_tokens(user_text, model)
        except Exception as e:
            print(f"Warning: Token counting failed - {e}")

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            end_time = time.time()

            result = response.json()
            debug_str = json.dumps(result, default=str)
            print(f"API response ({len(debug_str)} chars): {debug_str[:500]}")

            if not result.get("choices") or not result["choices"][0].get("message"):
                raise ValueError("Invalid response format from API: 'choices' or 'message' missing.")

            message = result["choices"][0]["message"]
            text_output = message.get("content", "")
            image_tensor = placeholder_image

            if message.get("images"):
                print(f"Found {len(message['images'])} image(s) in API response")
                try:
                    first_image = message["images"][0]
                    image_url = first_image["image_url"]["url"]

                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",", 1)[1]
                        try:
                            image_tensor = self.base64_to_image(base64_str)
                            print(f"Successfully decoded image from API response")
                        except Exception as e:
                            print(f"Error decoding image: {e}")
                    else:
                        print(f"Image URL format not supported: {image_url[:50]}...")
                except Exception as e:
                    print(f"Error processing images from response: {e}")
            else:
                print("No images found in API response")

            if isinstance(text_output, list):
                text_parts = []
                for content in text_output:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            text_parts.append(content.get("text", ""))
                        elif content.get("type") == "image_url":
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                base64_str = image_url.split(",", 1)[1]
                                try:
                                    image_tensor = self.base64_to_image(base64_str)
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                text_output = "\n".join(text_parts)

            response_ms = result.get("response_ms", None)
            api_usage = result.get("usage", {})
            prompt_tokens = api_usage.get("prompt_tokens", text_token_estimate)
            completion_tokens = api_usage.get("completion_tokens", 0)

            tps = 0
            elapsed_time = end_time - start_time
            if response_ms is not None:
                server_elapsed_time = response_ms / 1000.0
                if server_elapsed_time > 0 and completion_tokens > 0:
                    tps = completion_tokens / server_elapsed_time
            elif elapsed_time > 0 and completion_tokens > 0:
                tps = completion_tokens / elapsed_time

            stats_text = (
                f"TPS: {tps:.2f}, "
                f"Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, "
                f"Temp: {validated_temp:.1f}, "
                f"Model: {modified_model}, "
                f"Image Generation: enabled"
            )

            credits_text = self.fetch_credits(api_key)

            return (text_output, image_tensor, placeholder_video, stats_text, credits_text, "")

        except requests.exceptions.RequestException as e:
            error_message = f"API Request Error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_message += f" | Details: {error_detail}"
                except json.JSONDecodeError:
                    error_message += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}"
            else:
                error_message += " (Network or connection issue)"
            print(f"ERROR: {error_message}")
            return (error_message, placeholder_image, placeholder_video, "Stats N/A due to error", "Credits N/A due to error", "")
        except Exception as e:
            print(f"ERROR: Node Error: {str(e)}")
            return (f"Node Error: {str(e)}", placeholder_image, placeholder_video, "Stats N/A due to error", "Credits N/A due to error", "")

    def _generate_video(
        self,
        api_key,
        model,
        seed,
        video_mode,
        video_prompt,
        video_resolution,
        video_aspect_ratio,
        duration,
        generate_audio,
        poll_interval_seconds,
        timeout_seconds,
        provider_json,
        video_frame_1=None,
        video_frame_2=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
    ):
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        if not api_key.strip():
            raise ValueError("api_key is required.")

        warnings: List[str] = []
        payload = self._build_payload(
            model=model,
            mode=video_mode,
            prompt=video_prompt,
            resolution=video_resolution,
            aspect_ratio=video_aspect_ratio,
            duration=duration,
            generate_audio=generate_audio,
            seed=seed,
            provider_json=provider_json,
            warnings=warnings,
            image_1=video_frame_1,
            image_2=video_frame_2,
            reference_image_1=reference_image_1,
            reference_image_2=reference_image_2,
            reference_image_3=reference_image_3,
            reference_image_4=reference_image_4,
        )
        model_data = OpenRouterCatalog.get_video_model_by_id(model)
        estimated_cost = OpenRouterCatalog.estimate_video_cost(model_data, payload, video_mode)

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
                "mode": video_mode,
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
        return ("", placeholder_image, video_output, status_text, "", warnings_text)

    @classmethod
    def IS_CHANGED(cls, api_key, request_type, model, seed,
                   system_prompt, user_message_box, image_generation_only,
                   web_search, cheapest, fastest, aspect_ratio,
                   image_resolution, temperature, pdf_engine, chat_mode,
                   video_mode, video_prompt, video_resolution,
                   video_aspect_ratio, duration, generate_audio,
                   poll_interval_seconds, timeout_seconds, provider_json,
                   pdf_data=None, user_message_input=None,
                   video_frame_1=None, video_frame_2=None,
                   reference_image_1=None, reference_image_2=None,
                   reference_image_3=None, reference_image_4=None,
                   **kwargs):
        image_hashes = []
        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')],
                           key=lambda x: int(x.split('_')[1]))

        for image_key in image_keys:
            if kwargs[image_key] is not None:
                image = kwargs[image_key]
                if isinstance(image, torch.Tensor):
                    try:
                        hasher = hashlib.sha256()
                        hasher.update(image.cpu().numpy().tobytes())
                        image_hashes.append(hasher.hexdigest())
                    except Exception as e:
                        print(f"Warning: Could not hash {image_key} data for IS_CHANGED: {e}")
                        image_hashes.append(f"{image_key}_hashing_error")
                else:
                    image_hashes.append(None)

        video_frame_hashes = []
        for image in [video_frame_1, video_frame_2, reference_image_1, reference_image_2, reference_image_3, reference_image_4]:
            if isinstance(image, torch.Tensor):
                try:
                    hasher = hashlib.sha256()
                    hasher.update(image.cpu().numpy().tobytes())
                    video_frame_hashes.append(hasher.hexdigest())
                except Exception:
                    video_frame_hashes.append("hashing_error")
            else:
                video_frame_hashes.append(None)

        pdf_hash = None
        if pdf_data is not None and isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
            try:
                hasher = hashlib.sha256()
                hasher.update(pdf_data["bytes"])
                pdf_hash = hasher.hexdigest()
            except Exception as e:
                print(f"Warning: Could not hash pdf data for IS_CHANGED: {e}")
                pdf_hash = "pdf_hashing_error"
        elif pdf_data is not None:
            pdf_hash = "invalid_pdf_data_format"

        try:
            temp_float = float(temperature) if isinstance(temperature, (str, int, float)) else 1.0
            temp_float = max(0.0, min(2.0, temp_float))
        except (ValueError, TypeError):
            temp_float = 1.0

        return (api_key, request_type, model, seed,
                system_prompt, user_message_box, image_generation_only,
                web_search, cheapest, fastest, aspect_ratio,
                image_resolution, temp_float, pdf_engine, chat_mode,
                video_mode, video_prompt, video_resolution,
                video_aspect_ratio, duration, generate_audio,
                poll_interval_seconds, timeout_seconds, provider_json,
                tuple(image_hashes), tuple(video_frame_hashes), pdf_hash, user_message_input)


NODE_CLASS_MAPPINGS = {
    "OpenRouterNode": OpenRouterNode,
    "openrouter_node": OpenRouterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterNode": "OpenRouter (Chat / Image / Video)",
    "openrouter_node": "OpenRouter (Chat / Image / Video)",
}
