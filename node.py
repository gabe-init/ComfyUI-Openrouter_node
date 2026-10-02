import requests
import json
import time
import base64
import io
import numpy as np
import torch
import tiktoken
from PIL import Image
import hashlib # Added for hashing PDF bytes in IS_CHANGED
import os
import re
import logging
from .chat_manager import ChatSessionManager

# Define a placeholder type name for PDF data.
# The actual input connection will accept '*' but we check the structure.
# Expecting a dictionary: {"filename": str, "bytes": bytes}
PDF_DATA_TYPE = "*" # Use '*' to accept any type, check structure later

class OpenRouterNode:
    """
    OpenRouter chat, image, and video requests with optional media inputs.
    Preserves Output, image, Stats, Credits and appends native VIDEO.
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600  # Cache duration in seconds (1 hour)
    default_request_timeout = 120
    min_request_timeout = 1
    max_request_timeout = 3600
    reasoning_effort_options = ("auto", "none", "minimal", "low", "medium", "high", "xhigh")
    default_reasoning_effort = "auto"

    def __init__(self):
        self.chat_manager = ChatSessionManager()
        self.last_video_job_id = ""

    @staticmethod
    def get_api_key(api_key_ui):
        """
        Resolves the API key from:
        1. UI input field (if not empty)
        2. Environment variable 'LLM_KEY'
        3. config file 'openrouter_api_key.json' in node directory
        """
        if api_key_ui and api_key_ui.strip():
            return api_key_ui.strip()

        # Check environment variable
        env_key = os.environ.get("LLM_KEY")
        if env_key and env_key.strip():
            return env_key.strip()

        # Check JSON file
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openrouter_api_key.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    file_key = config.get("api_key")
                    if file_key and file_key.strip():
                        return file_key.strip()
            except Exception as e:
                print(f"Error reading openrouter_api_key.json: {e}")

        return ""

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input specification for this node.
        Includes optional inputs for image and PDF data.
        """
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "user_message_box": ("STRING", {
                    "multiline": True,
                    "default": "Hello, how are you?"
                }),
                "model": (cls.fetch_openrouter_models(),),
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
                "image_resolution": (["auto", "0.5K", "1K", "2K", "4K"], {"default": "auto"}),
                "reasoning_effort": (list(cls.reasoning_effort_options), {"default": cls.default_reasoning_effort}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
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
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "pdf_data": (PDF_DATA_TYPE,), # Use '*' and check structure in generate_response
                "user_message_input": ("STRING", {"forceInput": True}),
                "audio_data": ("AUDIO",),
                "request_type": (["chat", "image", "video"], {"default": "chat"}),
                "service_tier": (["auto", "default", "flex", "priority", "ultrafast"], {"default": "auto"}),
                "image_quality": (["auto", "low", "medium", "high"], {"default": "auto"}),
                "image_background": (["auto", "opaque", "transparent"], {"default": "auto"}),
                "video_mode": (["text_to_video", "first_frame", "first_last_frame", "reference_images"], {"default": "text_to_video"}),
                "video_duration": ("STRING", {"default": "auto"}),
                "video_resolution": ("STRING", {"default": "auto"}),
                "video_generate_audio": ("BOOLEAN", {"default": False}),
                "video_wait_timeout": ("INT", {"default": 900, "min": 1, "max": 3600}),
                "video_job_id": ("STRING", {"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("Output", "image", "Stats", "Credits", "video")

    FUNCTION = "generate_response"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """Read the cached union without blocking ComfyUI's schema/event loop."""
        from . import openrouter_catalog
        snapshot = openrouter_catalog.get_catalog()
        ids = {m["id"] for kind in ("chat", "image", "video")
               for m in snapshot.get(kind, []) if isinstance(m.get("id"), str)}
        return sorted(ids) or ["openai/gpt-4o"]

    @classmethod
    def VALIDATE_INPUTS(cls, model, aspect_ratio="auto", image_resolution="auto",
                        image_quality="auto", image_background="auto"):
        # Discovery is asynchronous. Do not reject a saved/manual model merely
        # because the server's widget snapshot predates a catalog refresh.
        return True

    def validate_temperature(self, temperature):
        """
        Validates and converts temperature value to float within acceptable range.
        """
        try:
            temp = float(temperature)
            return max(0.0, min(2.0, temp))  # Clamp between 0.0 and 2.0
        except (ValueError, TypeError):
            return 1.0  # Return default if conversion fails

    def validate_request_timeout(self, request_timeout):
        """
        Validates and converts request timeout to seconds within an acceptable range.
        """
        try:
            timeout = int(request_timeout)
            return max(self.min_request_timeout, min(self.max_request_timeout, timeout))
        except (ValueError, TypeError):
            return self.default_request_timeout

    @classmethod
    def validate_reasoning_effort(cls, reasoning_effort):
        """
        Validates OpenRouter reasoning effort. "auto" means do not send a
        reasoning override and let OpenRouter/model defaults apply.
        """
        if isinstance(reasoning_effort, str):
            normalized_effort = reasoning_effort.strip().lower()
            if normalized_effort in cls.reasoning_effort_options:
                return normalized_effort
        return cls.default_reasoning_effort

    def fetch_credits(self, api_key, timeout=None):
        """
        Fetches the user's credits information from the OpenRouter API.
        Returns a formatted string with remaining credits.
        """
        api_key = self.get_api_key(api_key)
        if not api_key:
             return "API Key not provided."

        url = "https://openrouter.ai/api/v1/credits"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gabe-init/ComfyUI-Openrouter_node",
            "X-Title": "ComfyUI OpenRouter LLM Node",
        }

        try:
            validated_timeout = self.validate_request_timeout(timeout)
            response = requests.get(url, headers=headers, timeout=validated_timeout)
            response.raise_for_status()

            result = response.json()
            # Check if 'data' and expected keys exist
            if "data" in result and "total_credits" in result["data"] and "total_usage" in result["data"]:
                total_credits = result["data"]["total_credits"]
                total_usage = result["data"]["total_usage"]
                remaining = total_credits - total_usage
                credits_text = f"Remaining: ${remaining:.3f}"
            else:
                credits_text = "Could not parse credit data from response."

            return credits_text

        except requests.exceptions.RequestException as e:
            # Provide more context about the error
            error_message = f"Error fetching credits: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                 error_message += f" | Status Code: {e.response.status_code}"
            return error_message
        except json.JSONDecodeError:
             return "Error fetching credits: Could not decode JSON response."

    @staticmethod
    def _safe_error(error, api_key=""):
        message = str(error)
        if api_key:
            message = message.replace(api_key, "[redacted]")
        return re.sub(r"data:[^\s\"']+", "[media omitted]", message)[:600]

    def _remember_video_job(self, job_id, unique_id=None):
        """Expose recovery before polling, including when Comfy hides cancellation errors."""
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", job_id):
            raise ValueError("OpenRouter returned an invalid video job ID.")
        self.last_video_job_id = job_id
        logger = logging.getLogger(__name__)
        logger.info("OpenRouter video job: %s. Resume with video_job_id=%s.", job_id, job_id)
        if unique_id is None:
            return
        try:
            from server import PromptServer
            server = PromptServer.instance
            server.send_sync("openrouter.video_job", {"node_id": str(unique_id), "job_id": job_id},
                             sid=getattr(server, "client_id", None))
        except Exception:
            # UI delivery must never turn an accepted paid job into a failure.
            logger.warning("OpenRouter video job: %s. Resume with video_job_id=%s; frontend update unavailable.",
                           job_id, job_id)

    @staticmethod
    def _chat_capabilities(model):
        """Wait for discovery once; retain catalog variants such as :free."""
        from .openrouter_catalog import get_model, require_model
        try:
            return require_model("chat", model)
        except ValueError as error:
            base_model = model
            while ":" in base_model and base_model.rsplit(":", 1)[1] in {"floor", "nitro", "online"}:
                base_model = base_model.rsplit(":", 1)[0]
            if base_model != model:
                # The blocking lookup has finished; use its refreshed snapshot.
                try:
                    return get_model("chat", base_model)
                except ValueError:
                    pass
            raise error

    @staticmethod
    def _chat_image_config(model, aspect_ratio, image_resolution):
        """Validate chat image controls using the matching image catalog record."""
        from .openrouter_catalog import get_model
        ratio = str(aspect_ratio).split(" ", 1)[0]
        resolution = "512" if image_resolution == "0.5K" else image_resolution
        try:
            record = get_model("image", model)
            parameters = record.get("supported_parameters") or {}
        except ValueError:
            parameters = {}

        def supports(name, value):
            descriptor = parameters.get(name)
            return isinstance(descriptor, dict) and descriptor.get("type") == "enum" and value in descriptor.get("values", [])

        config = {}
        if ratio != "auto":
            if not supports("aspect_ratio", ratio):
                raise ValueError(f"Cannot verify aspect ratio {ratio} for this chat image model; choose auto or use the Image API.")
            config["aspect_ratio"] = ratio
        # The old node always stored 1K, including for GPT image models. It is
        # neutral when unavailable; explicit non-default choices must validate.
        if resolution == "1K" and not supports("resolution", resolution):
            resolution = "auto"
        if resolution != "auto":
            if not model.startswith("google/gemini-") or not supports("resolution", resolution):
                raise ValueError(f"Resolution {resolution} is not supported for this chat image model; choose auto or use the Image API.")
            config["image_size"] = resolution
        return config

    def generate_response(self, api_key, system_prompt, user_message_box, model,
                          web_search, cheapest, fastest, temperature, pdf_engine, chat_mode,
                          request_timeout=120, aspect_ratio="auto", image_resolution="1K", seed=0,
                          pdf_data=None, user_message_input=None, reasoning_effort="auto",
                          audio_data=None, request_type="chat", service_tier="auto",
                          image_quality="auto", image_background="auto", video_mode="text_to_video",
                          video_duration="auto", video_resolution="auto", video_generate_audio=False,
                          video_wait_timeout=900, video_job_id="", unique_id=None, **kwargs):
        """Keep the original four output positions; append native video output."""
        placeholder = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        key = self.get_api_key(api_key)
        self.last_video_job_id = video_job_id.strip() if request_type == "video" else ""
        try:
            if not key:
                raise ValueError("API Key not provided. Set LLM_KEY or openrouter_api_key.json.")
            if request_type not in ("chat", "image", "video"):
                raise ValueError("request_type must be chat, image, or video")
            if service_tier not in ("auto", "default", "flex", "priority", "ultrafast"):
                raise ValueError("Unsupported service tier")
            timeout = self.validate_request_timeout(request_timeout)
            prompt = user_message_input if user_message_input is not None else user_message_box
            ratio = str(aspect_ratio).split(" ", 1)[0]
            if request_type == "chat":
                if pdf_data is not None and (not isinstance(pdf_data, dict) or
                        not isinstance(pdf_data.get("bytes"), bytes) or not pdf_data["bytes"]):
                    raise ValueError("pdf_data must contain nonempty PDF bytes")
                has_images = any(value is not None for name, value in kwargs.items() if re.fullmatch(r"image_\d+", name))
                # 1K was an unconditional saved value in older text workflows.
                image_controls = ratio != "auto" or image_resolution not in ("auto", "1K")
                try:
                    capabilities = self._chat_capabilities(model)
                except ValueError as error:
                    if audio_data is not None or has_images or image_controls:
                        raise ValueError("Cannot verify the selected chat model's media capabilities. Refresh models before submitting audio, images, or image settings.") from error
                    capabilities = {}  # Plain custom chat IDs may be absent from discovery.
                architecture = capabilities.get("architecture") or {}
                inputs = architecture.get("input_modalities")
                for modality, present in (("audio", audio_data is not None), ("image", has_images)):
                    if present and not isinstance(inputs, list):
                        raise ValueError(f"Cannot verify whether the selected chat model accepts {modality} input.")
                    if present and modality not in inputs:
                        raise ValueError(f"The selected chat model does not support {modality} input")
                if image_controls and "image" not in architecture.get("output_modalities", []):
                    raise ValueError("The selected chat model does not advertise image output for these image settings.")
                audio_content = None
                if audio_data is not None:
                    from .openrouter_audio import prepare_audio
                    audio_content = prepare_audio(audio_data)["block"]
                result = self._generate_chat(
                    key, system_prompt, user_message_box, model, web_search, cheapest, fastest,
                    temperature, pdf_engine, chat_mode, request_timeout=timeout,
                    aspect_ratio=aspect_ratio, image_resolution=image_resolution, seed=seed,
                    pdf_data=pdf_data, user_message_input=user_message_input,
                    reasoning_effort=reasoning_effort, service_tier=service_tier,
                    audio_content=audio_content, capabilities=capabilities, **kwargs)
                return (*result, None)

            # Recovery must not process or validate inputs from the original job.
            resuming = request_type == "video" and bool(video_job_id.strip())
            references = []
            if not resuming:
                if audio_data is not None or pdf_data is not None:
                    raise ValueError("Audio and PDF inputs are supported in chat mode only.")
                for name in sorted((k for k in kwargs if re.fullmatch(r"image_\d+", k)),
                                   key=lambda k: int(k.split("_")[1])):
                    if kwargs[name] is not None:
                        references.append("data:image/png;base64," + self.image_to_base64(kwargs[name]))
            if request_type == "image":
                from .openrouter_images import generate_images
                from .openrouter_catalog import require_model
                record = require_model("image", model)
                resolution = image_resolution
                # Older workflows always stored 1K even when the model did not
                # support resolution. Do not turn that inherited default into a
                # new unsupported /images parameter.
                if resolution == "1K" and "resolution" not in record.get("supported_parameters", {}):
                    resolution = "auto"
                result = generate_images(
                    key, model, prompt, reference_urls=references, aspect_ratio=ratio,
                    resolution=resolution, quality=image_quality, background=image_background,
                    seed=seed, timeout=timeout)
                tensors = []
                for raw in result["images"]:
                    with Image.open(io.BytesIO(raw)) as img:
                        mode = "RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB"
                        array = np.asarray(img.convert(mode), dtype=np.float32) / 255.0
                        tensors.append(torch.from_numpy(array.copy()).unsqueeze(0))
                if not tensors:
                    raise ValueError("OpenRouter returned no raster images")
                image = torch.cat(tensors, dim=0)
                video = None
            else:
                from .openrouter_video import generate_video
                self.last_video_job_id = video_job_id.strip()
                result = generate_video(
                    key, model, prompt, reference_urls=references, mode=video_mode,
                    duration=video_duration, resolution=video_resolution, aspect_ratio=ratio,
                    generate_audio=video_generate_audio, seed=seed, request_timeout=timeout,
                    wait_timeout=video_wait_timeout, job_id=video_job_id,
                    on_job=lambda job: self._remember_video_job(job, unique_id))
                image, video = placeholder, result["video"]
            usage = result.get("usage") or {}
            stats = f"Model: {model}"
            cost = result.get("cost", usage.get("cost"))
            if isinstance(cost, (int, float)):
                stats += f", Cost: ${cost:.6f}"
            if result.get("job_id"):
                stats += f", Job: {result['job_id']}"
            return (result.get("text", ""), image, stats, self.fetch_credits(key, timeout=timeout), video)
        except Exception as error:
            message = self._safe_error(error, key)
            if request_type == "video" and self.last_video_job_id:
                if f"video_job_id={self.last_video_job_id}" not in message:
                    message += f" | resume with video_job_id={self.last_video_job_id}"
            if request_type == "video":
                # A None VIDEO causes a misleading error in downstream SaveVideo
                # and caches the failed generation as a successful node result.
                failure = RuntimeError(f"OpenRouter video error: {message}")
                failure.openrouter_job_id = self.last_video_job_id
                raise failure from None
            return (f"Error: {message}", placeholder, "Stats N/A due to error", "Credits N/A due to error", None)

    def _generate_chat(self, api_key, system_prompt, user_message_box, model,
                         web_search, cheapest, fastest, temperature, pdf_engine, chat_mode,
                         request_timeout=120, aspect_ratio="auto", image_resolution="1K", seed=0,
                         pdf_data=None, user_message_input=None, reasoning_effort="auto",
                         service_tier="auto", audio_content=None, capabilities=None, **kwargs):
        """
        Sends a completion request to the OpenRouter chat completion endpoint.
        Handles text, optional image, and optional PDF inputs.

        Returns four outputs:
          (1) Output: the LLM's text response
          (2) image: an image tensor if the response contains an image, else empty tensor
          (3) Stats: a string with tokens per second, prompt tokens, completion tokens
          (4) Credits: a string with the user's credit information
        """
        # Create empty placeholder image
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        
        # Resolve API key
        api_key = self.get_api_key(api_key)
        
        if not api_key:
             return ("Error: API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json", placeholder_image, "Stats N/A", "Credits N/A")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gabe-init/ComfyUI-Openrouter_node",
            "X-Title": "ComfyUI OpenRouter LLM Node",
        }

        # Validate and convert temperature
        validated_temp = self.validate_temperature(temperature)
        validated_timeout = self.validate_request_timeout(request_timeout)
        validated_reasoning_effort = self.validate_reasoning_effort(reasoning_effort)

        # Decide whether to use user_message_input or user_message_box
        user_text = user_message_input if user_message_input is not None else user_message_box

        # Initialize session_path
        session_path = None
        
        # Handle chat mode
        if chat_mode:
            # Get or create a chat session
            session_path, messages = self.chat_manager.get_or_create_session(user_text, system_prompt)
            
            # Check if we need to update the system prompt (for existing sessions)
            if messages and messages[0]["role"] == "system" and messages[0]["content"] != system_prompt:
                # Update system prompt if it has changed
                messages[0]["content"] = system_prompt
        else:
            # Non-chat mode: Build the messages array, starting with a system prompt.
            messages = [
                {"role": "system", "content": system_prompt},
            ]

        # --- Build the user message content ---
        user_content_blocks = []

        # 1. Add Text part (always present)
        user_content_blocks.append({
            "type": "text",
            "text": user_text
        })
        if audio_content is not None:
            user_content_blocks.append(audio_content)

        # 2. Add Image parts (optional) - support multiple images from kwargs
        # Process all image_N inputs from kwargs
        image_keys = sorted([k for k in kwargs if re.fullmatch(r'image_\d+', k)],
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
                    return (f"Error processing {image_key}: {e}", placeholder_image, "Stats N/A", "Credits N/A")

        # 3. Add PDF part (optional)
        pdf_filename = "document.pdf" # Default filename if not provided
        if pdf_data is not None:
            # Validate pdf_data structure (expecting dict with 'filename' and 'bytes')
            if isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
                pdf_bytes = pdf_data["bytes"]
                # Use provided filename if available and valid, otherwise use default
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
                    return (f"Error encoding PDF: {e}", placeholder_image, "Stats N/A", "Credits N/A")
            else:
                # Handle case where pdf_data is not in the expected format
                print(f"Warning: pdf_data input is not in the expected format (dict with 'filename' and 'bytes'). PDF not included.")
                # Optionally return an error or just proceed without the PDF
                # return ("Error: Invalid PDF data format.", "Stats N/A", "Credits N/A")


        # Determine message format based on content type
        # Use simple string format for text-only requests to ensure compatibility
        # Use structured format only when we have multimodal content
        has_multimodal_content = len(user_content_blocks) > 1 or any(block.get("type") != "text" for block in user_content_blocks)
        
        if has_multimodal_content:
            # Use structured format for multimodal content
            new_user_message = {
                "role": "user",
                "content": user_content_blocks
            }
        else:
            # Use simple string format for text-only requests
            new_user_message = {
                "role": "user",
                "content": user_text
            }
        
        if chat_mode:
            # In chat mode, append to existing conversation (but don't save yet - wait for response)
            messages.append(new_user_message)
        else:
            # In non-chat mode, messages array already has system prompt, just append user message
            messages.append(new_user_message)

        # --- Apply model modifiers ---
        modified_model = model
        # Check if model already has modifiers to avoid duplication
        if web_search and ":online" not in modified_model:
            modified_model = f"{modified_model}:online"
        if ":online" not in modified_model:
             if cheapest and ":floor" not in modified_model:
                 modified_model = f"{modified_model}:floor"
             elif fastest and not cheapest and ":nitro" not in modified_model:
                 modified_model = f"{modified_model}:nitro"


        # --- Construct the final payload ---
        data = {
            "model": modified_model,
            "messages": messages,
            "temperature": validated_temp,
            "seed": seed
        }
        if validated_reasoning_effort != "auto":
            data["reasoning"] = {"effort": validated_reasoning_effort}
        if service_tier != "auto":
            data["service_tier"] = service_tier

        capabilities = capabilities or {}
        output_modalities = capabilities.get("architecture", {}).get("output_modalities", [])
        if "image" in output_modalities:
            data["modalities"] = [m for m in ("text", "image") if m in output_modalities]
            config = self._chat_image_config(capabilities.get("id", model), aspect_ratio, image_resolution)
            if config:
                data["image_config"] = config

        print(f"Payload: model={modified_model}")

        # Add plugins if a specific PDF engine is selected
        if pdf_engine != "auto":
             data["plugins"] = [
                 {
                     "id": "file-parser",
                     "pdf": {
                         "engine": pdf_engine
                     }
                 }
             ]

        # --- Pre-calculate text input tokens (rough estimate) ---
        # Note: Actual token count depends on the model and includes parsed PDF/image data.
        # Rely on the API response for accurate usage stats.
        text_token_estimate = 0
        try:
            text_token_estimate = self.count_tokens(system_prompt, model) + self.count_tokens(user_text, model)
        except Exception as e:
            print(f"Warning: Token counting failed - {e}")


        # --- Make API Call and Process Response ---
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=validated_timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            end_time = time.time()

            result = response.json()

            # --- Extract results and calculate stats ---
            if not result.get("choices") or not result["choices"][0].get("message"):
                 raise ValueError("Invalid response format from API: 'choices' or 'message' missing.")

            # Parse response for text and image content
            message = result["choices"][0]["message"]
            text_output = message.get("content") or ""
            image_tensor = placeholder_image

            # Check for images in the separate images field (OpenRouter format)
            if message.get("images"):
                print(f"Found {len(message['images'])} image(s) in API response")
                try:
                    # Get the first image from the images array
                    first_image = message["images"][0]
                    image_url = first_image["image_url"]["url"]
                    
                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",", 1)[1]
                        try:
                            # Convert base64 to image tensor
                            image_tensor = self.base64_to_image(base64_str)
                            print(f"Successfully decoded image from API response")
                        except Exception as e:
                            print(f"Error decoding image: {e}")
                    else:
                        raise ValueError("Chat image response must contain encoded image data.")
                except Exception as e:
                    print(f"Error processing images from response: {e}")
            else:
                print("No images found in API response - this may be normal if the model doesn't support image generation or the prompt didn't request an image")
            
            # Also handle legacy multimodal content format as fallback
            if isinstance(text_output, list):
                text_parts = []
                for content in text_output:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            text_parts.append(content.get("text", ""))
                        elif content.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                base64_str = image_url.split(",", 1)[1]
                                try:
                                    # Convert base64 to image tensor
                                    image_tensor = self.base64_to_image(base64_str)
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                text_output = "\n".join(text_parts)

            response_ms = result.get("response_ms", None)
            api_usage = result.get("usage", {})
            prompt_tokens = api_usage.get("prompt_tokens", text_token_estimate) # Use API value if available
            completion_tokens = api_usage.get("completion_tokens", 0)
            if completion_tokens == 0 and text_output: # Estimate completion tokens if API doesn't provide them
                 try:
                     completion_tokens = self.count_tokens(text_output, model)
                 except Exception as e:
                     print(f"Warning: Completion token counting failed - {e}")


            # Calculate tokens per second (TPS)
            tps = 0
            elapsed_time = end_time - start_time
            if response_ms is not None:
                server_elapsed_time = response_ms / 1000.0
                if server_elapsed_time > 0:
                    tps = completion_tokens / server_elapsed_time
            elif elapsed_time > 0:
                # Use client-side timing as fallback, less accurate due to network latency
                 tps = completion_tokens / elapsed_time
                 # Optional: apply a heuristic correction factor if needed, but server time is better
                 # correction_factor = 1.28 # Example factor, might need tuning
                 # tps *= correction_factor

            stats_text = (
                f"TPS: {tps:.2f}, "
                f"Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, "
                f"Temp: {validated_temp:.1f}, "
                f"Model: {modified_model}" # Display the actual model used
            )
            if pdf_engine != "auto":
                 stats_text += f", PDF Engine: {pdf_engine}"
            if validated_reasoning_effort != "auto":
                 stats_text += f", Reasoning: {validated_reasoning_effort}"
            if result.get("service_tier"):
                stats_text += f", Service tier: {result['service_tier']}"
            if isinstance(api_usage.get("cost"), (int, float)):
                stats_text += f", Cost: ${api_usage['cost']:.6f}"


            # Fetch credits information AFTER the main request
            credits_text = self.fetch_credits(api_key, timeout=validated_timeout)

            # Save conversation in chat mode
            if chat_mode and session_path:
                # Append assistant's response to the conversation
                assistant_message = {
                    "role": "assistant",
                    "content": text_output
                }
                messages.append(assistant_message)
                
                # Save the updated conversation
                self.chat_manager.save_conversation(session_path, messages)

            return (text_output, image_tensor, stats_text, credits_text)

        except requests.exceptions.RequestException as e:
            error_message = f"API Request Error: {self._safe_error(e, api_key)}"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
            else:
                 error_message += " (Network or connection issue)"
            print(f"ERROR: {error_message}")
            return (error_message, placeholder_image, "Stats N/A due to error", "Credits N/A due to error")
        except Exception as e:
             print(f"ERROR: Node Error: {str(e)}")
             return (f"Node Error: {str(e)}", placeholder_image, "Stats N/A due to error", "Credits N/A due to error")

    @staticmethod
    def image_to_base64(image):
        """
        Converts a ComfyUI IMAGE (torch.Tensor, BHWC, float 0-1)
        into a base64-encoded PNG string.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input 'image' is not a torch.Tensor")

        # Remove batch dimension if present
        if image.ndim == 4:
            if image.shape[0] != 1:
                 raise ValueError("Connect one image per numbered input; image batches are not supported.")
            image = image.squeeze(0) # Shape HWC

        if image.ndim != 3:
             raise ValueError(f"Unexpected image dimensions: {image.shape}. Expected HWC.")

        # Convert float tensor (0-1) to numpy array (0-255, uint8)
        image_np = image.detach().cpu().float().numpy()
        if not np.isfinite(image_np).all():
            raise ValueError("Image contains non-finite values")
        if image_np.dtype != np.uint8:
             if image_np.min() < 0 or image_np.max() > 1:
                  print("Warning: Image tensor values outside [0, 1] range. Clamping.")
                  image_np = np.clip(image_np, 0, 1)
             image_np = (image_np * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        if image_np.shape[-1] not in (3, 4):
            raise ValueError("Image must contain RGB or RGBA channels")
        pil_image = Image.fromarray(image_np)

        # Save PIL Image to a bytes buffer as PNG
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")

        # Encode the bytes buffer to base64 string
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def base64_to_image(base64_str: str) -> torch.Tensor:
        """
        Converts a base64 image string to a ComfyUI image tensor
        Returns tensor in [1, H, W, 3] format with values in [0, 1]
        """
        try:
            # Decode base64 string to image
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            img = img.convert("RGB")

            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Add batch dimension: [1, H, W, 3]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            print(f"Successfully converted base64 to image tensor: {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            print(f"Error in base64_to_image: {e}")
            # Return a small placeholder image instead of failing
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    @staticmethod
    def count_tokens(text, model):
        """
        Count tokens for a given text using tiktoken.
        Uses model-specific encodings where possible, falls back to cl100k_base.
        Handles potential errors during encoding.
        """
        if not text or not isinstance(text, str):
            return 0

        # Strip any model modifiers like :floor, :nitro, :online
        base_model = model.split(':')[0] if ':' in model else model

        # Simplified mapping, cl100k_base is common for many recent models
        encoding_name = "cl100k_base"
        try:
            # List known models/prefixes that definitely use cl100k_base
            # Add others if known, but cl100k_base is a safe default for many
            cl100k_models = [
                "openai/gpt-4", "openai/gpt-3.5", "openai/gpt-4o",
                "anthropic/claude",
                "google/gemini",
                "meta-llama/llama-2", "meta-llama/llama-3",
                "mistralai/mistral", "mistralai/mixtral",
            ]
            # Check if the base_model or its prefix matches known cl100k models
            is_cl100k = any(base_model.startswith(prefix) for prefix in cl100k_models)

            if is_cl100k:
                 encoding_name = "cl100k_base"
            # else: # Add logic for other encodings if needed, e.g., p50k_base for older models
            #    pass # Stick with cl100k_base as default for now

            encoding = tiktoken.get_encoding(encoding_name)
            token_count = len(encoding.encode(text, disallowed_special=())) # Allow special tokens
            return token_count

        except Exception as e:
            print(f"Warning: Tiktoken error for model '{model}' (base: '{base_model}', encoding: '{encoding_name}'): {e}. Falling back to estimation.")
            # Fallback: Estimate tokens based on characters (rough approximation)
            # Average ~4 chars per token is a common heuristic
            return max(1, round(len(text) / 4))


    @classmethod
    def IS_CHANGED(cls, api_key, system_prompt, user_message_box, model,
                   web_search, cheapest, fastest, temperature, pdf_engine, chat_mode,
                   request_timeout=120, aspect_ratio="auto", image_resolution="1K", seed=0,
                   pdf_data=None, user_message_input=None, reasoning_effort="auto",
                   audio_data=None, request_type="chat", service_tier="auto",
                   image_quality="auto", image_background="auto", video_mode="text_to_video",
                   video_duration="auto", video_resolution="auto", video_generate_audio=False,
                   video_wait_timeout=900, video_job_id="", **kwargs):
        """
        Check if any input that affects the output has changed.
        Includes hashing image and PDF data.
        """
        if request_type == "video" and video_job_id.strip():
            # Recovery must work even if original media disappeared or changed.
            return float("nan")  # Poll again after a previous local timeout.

        # Hash image data if present - handle multiple images from kwargs
        image_hashes = []
        image_keys = sorted([k for k in kwargs if re.fullmatch(r'image_\d+', k)],
                           key=lambda x: int(x.split('_')[1]))
        
        for image_key in image_keys:
            if kwargs[image_key] is not None:
                image = kwargs[image_key]
                if isinstance(image, torch.Tensor):
                    try:
                        hasher = hashlib.sha256()
                        normalized = image.detach().cpu().float().contiguous()
                        hasher.update(str(tuple(normalized.shape)).encode())
                        hasher.update(normalized.numpy().tobytes())
                        image_hashes.append(hasher.hexdigest())
                    except Exception as e:
                        print(f"Warning: Could not hash {image_key} data for IS_CHANGED: {e}")
                        raise ValueError(f"Could not hash {image_key}") from e
                else:
                    image_hashes.append(None)


        # Hash PDF data if present and valid
        pdf_hash = None
        if pdf_data is not None and isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
             try:
                 hasher = hashlib.sha256()
                 hasher.update(pdf_data["bytes"])
                 pdf_hash = hasher.hexdigest()
                 # Optionally include filename in hash if it affects processing?
                 # if "filename" in pdf_data: hasher.update(pdf_data["filename"].encode())
             except Exception as e:
                 print(f"Warning: Could not hash pdf data for IS_CHANGED: {e}")
                 pdf_hash = "pdf_hashing_error" # Use a placeholder on error
        elif pdf_data is not None:
             # Handle cases where pdf_data is present but not in the expected format
             pdf_hash = "invalid_pdf_data_format"


        # Ensure temperature is consistently represented (e.g., as float)
        try:
            temp_float = float(temperature) if isinstance(temperature, (str, int, float)) else 1.0
            temp_float = max(0.0, min(2.0, temp_float))
        except (ValueError, TypeError):
            temp_float = 1.0

        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        validated_reasoning_effort = cls.validate_reasoning_effort(reasoning_effort)


        # Combine all relevant inputs into a tuple for comparison
        # Use primitive types where possible for reliable hashing/comparison
        # Note: api_key here is the UI value only. Keys resolved from the LLM_KEY
        # env var or openrouter_api_key.json are intentionally NOT part of the
        # cache key â€” they're treated as user environment, not workflow inputs.
        audio_hash = None
        if audio_data is not None:
            from .openrouter_audio import audio_fingerprint
            audio_hash = audio_fingerprint(audio_data)
        return (hashlib.sha256(api_key.encode()).hexdigest(), system_prompt, user_message_box, model,
                web_search, cheapest, fastest, temp_float, pdf_engine, chat_mode,
                timeout_int, aspect_ratio, image_resolution, seed, validated_reasoning_effort,
                tuple(image_hashes), pdf_hash, user_message_input, audio_hash, request_type, service_tier,
                image_quality, image_background, video_mode, video_duration, video_resolution,
                video_generate_audio, video_wait_timeout, video_job_id)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "OpenRouterNode": OpenRouterNode,
    "openrouter_node": OpenRouterNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterNode": "OpenRouter LLM Node (Text/Multi-Image/PDF/Chat)" # Updated name
}
