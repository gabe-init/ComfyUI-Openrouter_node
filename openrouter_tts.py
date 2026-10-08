"""Dedicated OpenRouter TTS requests; no automatic retries of paid calls.

Uses OpenRouter's OpenAI-compatible /api/v1/audio/speech endpoint:
https://openrouter.ai/docs/guides/overview/multimodal/tts
"""

import math
import re
from urllib.parse import urlsplit

import requests

try:
    from . import openrouter_catalog as catalog
except ImportError:
    import openrouter_catalog as catalog


API_BASE = catalog.API_BASE + "/audio/speech"
DEFAULT_SAMPLE_RATE = 24000  # Documented default pcm rate (e.g. Azure: 24 kHz mono).
MAX_REFERENCE_CHARS = 10000  # Per-clip transcript limit enforced by OpenRouter.
# _frame_to_float normalizes PyAV sample formats to ComfyUI's [-1, 1] range.


def _redact(text, api_key):
    message = str(text)
    if isinstance(api_key, str) and api_key:
        message = message.replace(api_key, "[redacted]")
    return re.sub(r"data:[^\s\"']+", "[media omitted]", message)[:600]


def _require_metadata(model):
    """Wait for discovery once; tolerate routing variants like the chat path."""
    if __package__:
        from .openrouter_catalog import require_model
    else:
        from openrouter_catalog import require_model
    try:
        return require_model("speech", model)
    except ValueError as error:
        base_model = model
        while ":" in base_model and base_model.rsplit(":", 1)[1] in {"floor", "nitro", "online", "free"}:
            base_model = base_model.rsplit(":", 1)[0]
        if base_model != model:
            try:
                return require_model("speech", base_model)
            except ValueError:
                pass
        raise error


def _validate_voice(voice, metadata, model):
    """Reject unsupported voices before payment; presets cannot be verified."""
    if voice in (None, "", "auto"):
        supported = metadata.get("supported_voices") or []
        if supported:
            # Providers without a default voice reject an omitted voice with a
            # 400 ("An explicit voice is required"); fail locally with the
            # actionable list instead.
            preview = ", ".join(str(item) for item in supported[:8])
            more = " ..." if len(supported) > 8 else ""
            raise ValueError(
                f"'{model}' requires an explicit voice; it has no default. "
                f"Set tts_voice to one of: {preview}{more}."
            )
        return None
    if not isinstance(voice, str):
        raise ValueError("TTS voice must be a string.")
    voice = voice.strip()
    if not voice:
        supported = metadata.get("supported_voices") or []
        if supported:
            preview = ", ".join(str(item) for item in supported[:8])
            more = " ..." if len(supported) > 8 else ""
            raise ValueError(
                f"'{model}' requires an explicit voice; it has no default. "
                f"Set tts_voice to one of: {preview}{more}."
            )
        return None
    supported = metadata.get("supported_voices") or []
    if not supported:
        # Discovery lists no voices (e.g. a @preset). Submit and let the
        # provider reject a truly unsupported identifier.
        print(f"Warning: no voice list for '{model}'; sending voice='{voice}' unverified.")
        return voice
    if voice not in supported:
        preview = ", ".join(str(item) for item in supported[:8])
        more = " ..." if len(supported) > 8 else ""
        raise ValueError(
            f"'{model}' does not support voice '{voice}'. Supported voices: {preview}{more}."
        )
    return voice


def _reference_data_url(value):
    """Normalize one voice-cloning clip to a data: URL, mirroring chat audio."""
    if not isinstance(value, dict):
        raise ValueError("Voice references must be audio dicts (native AUDIO or file bytes).")
    if __package__:
        from .openrouter_audio import prepare_audio
    else:
        from openrouter_audio import prepare_audio
    prepared = prepare_audio(value)
    block = prepared["block"]["input_audio"]
    return f"data:audio/{block['format']};base64,{block['data']}"


def _reference_image_url(value):
    """Accept image tensors (node image inputs) or prebuilt URL strings."""
    if isinstance(value, str):
        if not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,", "http://", "https://")):
            raise ValueError("Image references must be PNG/JPEG/WebP data URLs or http(s) URLs.")
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https", "data"):
            raise ValueError("Image references must use http(s) or data URLs.")
        return value
    if not isinstance(value, dict) or "image_url" not in value:
        raise ValueError("Image references must be image tensors or prepared image_url parts.")
    url = value["image_url"].get("url")
    if not isinstance(url, str) or not url.startswith(("data:image/", "http://", "https://")):
        raise ValueError("Image references must use data: or http(s) URLs.")
    return url


def _turn(value, default_voice, default_instructions, index, metadata, model):
    """Validate one multi-speaker turn; accept dicts or raw strings."""
    if isinstance(value, dict):
        text = value.get("text")
        voice = _validate_voice(value.get("voice"), metadata, model) if value.get("voice") else None
        instructions = value.get("instructions")
    else:
        text, voice, instructions = value, None, None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Speaker turn {index} must contain nonempty text.")
    if instructions is not None and (not isinstance(instructions, str) or len(instructions) > MAX_REFERENCE_CHARS):
        raise ValueError(f"Speaker turn {index} instructions must be a string of at most {MAX_REFERENCE_CHARS} characters.")
    turn = {"text": text}
    # A turn without its own voice inherits the verified top-level voice.
    if voice is not None:
        turn["voice"] = voice
    elif default_voice is not None:
        turn["voice"] = default_voice
    if instructions:
        turn["instructions"] = instructions
    elif default_instructions:
        turn["instructions"] = default_instructions
    return turn


def build_payload(model, prompt, voice="auto", response_format="auto", instructions="",
                  speed=1.0, speaker_turns=None, reference_audio=None, reference_image=None,
                  reference_text=None, seed=0):
    """Validate all inputs and assemble the /audio/speech request body.

    Multi-speaker turns and references are mutually exclusive with plain text
    prompts on most providers; OpenRouter itself rejects unsupported mixes with
    a 400, which surfaces through the standard error path.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A TTS model is required.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A speech prompt (input text) is required.")
    if len(prompt) > 3000:
        raise ValueError(f"Speech input exceeds 3000 characters ({len(prompt)}); some models (Seed Audio) reject more than 3000.")
    if response_format not in (None, "", "auto", "mp3", "pcm"):
        raise ValueError("Audio response_format must be mp3 or pcm.")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError("Speech instructions must be a string.")
        if len(instructions) > MAX_REFERENCE_CHARS:
            raise ValueError(f"Speech instructions must be at most {MAX_REFERENCE_CHARS} characters.")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        raise ValueError("Speech speed must be a number between 0.25 and 4.0.")

    metadata = {}
    is_preset = model.startswith("@preset/")
    if not is_preset:
        metadata = _require_metadata(model) or {}
    voice = _validate_voice(voice, metadata, model)

    turns = []
    if speaker_turns:
        if not isinstance(speaker_turns, (list, tuple)):
            raise ValueError("Speaker turns must be a list of {text, voice, instructions} dicts.")
        if len(speaker_turns) > 10:
            raise ValueError("Multi-speaker input accepts at most 10 turns.")
        turns = [_turn(item, voice, instructions, index, metadata, model)
                 for index, item in enumerate(speaker_turns, start=1)]
        # OpenRouter resolves a turn voice from the top-level voice; sending the
        # top-level voice alongside per-turn voices would override turn choices.
        voice = None if any("voice" in turn for turn in turns) else voice

    references = []
    clips = [item for item in (reference_audio or []) if item is not None]
    if len(clips) > 3:
        raise ValueError("Voice cloning accepts at most 3 audio references.")
    if reference_text is not None:
        if not isinstance(reference_text, str) or len(reference_text) > MAX_REFERENCE_CHARS:
            raise ValueError(f"The reference transcript must be a string of at most {MAX_REFERENCE_CHARS} characters.")
        if not clips:
            raise ValueError("A reference transcript requires at least one reference clip.")
        if len(clips) > 1:
            raise ValueError("A reference transcript must immediately follow its clip; send one clip when providing a transcript.")
    for index, item in enumerate(clips):
        reference = {"type": "input_audio", "input_audio": {"data": _reference_data_url(item)}}
        references.append(reference)
        if reference_text and len(clips) == 1:
            # A single clip's transcript may follow it (OpenRouter TTS schema).
            references.append({"type": "text", "text": reference_text})
    if reference_image is not None:
        if clips:
            raise ValueError("Audio and image references cannot be combined in one request.")
        references.append({"type": "image_url", "image_url": {"url": _reference_image_url(reference_image)}})

    payload = {"model": model, "input": turns if turns else prompt}
    if voice is not None:
        payload["voice"] = voice
    if response_format not in (None, "", "auto"):
        payload["response_format"] = response_format
    if instructions:
        payload["instructions"] = instructions
    if speed != 1.0:
        payload["speed"] = float(speed)
    if references:
        payload["input_references"] = references
    return payload


def generate_speech(api_key, model, prompt, voice="auto", response_format="auto",
                    instructions="", speed=1.0, speaker_turns=None, reference_audio=None,
                    reference_image=None, reference_text=None, timeout=120, *, _http=None):
    """POST one paid request and decode the raw audio stream to a native AUDIO dict.

    Returns (audio_dict, generation_id). No automatic retries of paid calls.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("An OpenRouter API key is required for audio.")
    payload = build_payload(model, prompt, voice=voice, response_format=response_format,
                            instructions=instructions, speed=speed,
                            speaker_turns=speaker_turns, reference_audio=reference_audio,
                            reference_image=reference_image, reference_text=reference_text)
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/gabe-init/ComfyUI-Openrouter_node",
        "X-Title": "ComfyUI OpenRouter LLM Node",
    }
    http = _http or requests
    try:
        response = http.post(API_BASE, headers=headers, json=payload, timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        raise RuntimeError(f"Audio request failed ({type(exc).__name__}); no automatic retry was made. Check OpenRouter activity before retrying.") from None
    if not 200 <= response.status_code < 300:
        try:
            detail = response.json().get("error", f"HTTP {response.status_code}")
        except Exception:
            detail = f"HTTP {response.status_code}"
        raise RuntimeError(f"OpenRouter audio request failed ({response.status_code}): {_redact(detail, api_key)}")
    content_type = response.headers.get("Content-Type", "")
    generation_id = response.headers.get("X-Generation-Id", "")
    payload_bytes = response.content or b""
    if not isinstance(payload_bytes, (bytes, bytearray)):
        raise RuntimeError("OpenRouter returned a non-binary audio response.")
    audio = _decode_audio(bytes(payload_bytes), content_type)
    return audio, generation_id


def _pcm_to_tensor(pcm_bytes, sample_rate=DEFAULT_SAMPLE_RATE, channels=1):
    import numpy as np
    import torch

    if len(pcm_bytes) % 2:
        raise ValueError("OpenRouter returned truncated PCM audio (odd byte count).")
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1 and len(samples) % channels:
        raise ValueError("OpenRouter returned truncated PCM audio (incomplete frame).")
    waveform = torch.from_numpy(samples.reshape(-1, channels).T.copy()).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _frame_to_float(array, format_name):
    import numpy as np

    # PyAV delivers raw coded sample values; normalize to ComfyUI's [-1, 1] range.
    if format_name.startswith("u8"):
        return array.astype(np.float32) / 128.0 - 1.0
    if format_name.startswith("s16"):
        return array.astype(np.float32) / 32768.0
    if format_name.startswith("s32"):
        return array.astype(np.float32) / 2147483648.0
    if format_name.startswith(("flt", "dbl")):
        return array.astype(np.float32)
    raise ValueError(f"Unsupported audio sample format '{format_name}' in the audio response.")


def _decode_audio(payload, content_type):
    """Turn the raw response body into a native ComfyUI AUDIO dict."""
    content_type = (content_type or "").lower()
    if not payload:
        raise ValueError("OpenRouter returned an empty audio response.")
    if "json" in content_type:
        raise ValueError("OpenRouter returned an error document instead of audio.")
    if "pcm" in content_type:
        return _pcm_to_tensor(payload)
    # mp3/wav/opus and anything else containerized: decode via PyAV (ComfyUI ships it).
    return _decode_container(payload)


def _decode_container(payload):
    import io

    import torch

    try:
        import av
    except ImportError as exc:
        raise ValueError("Audio output requires PyAV (ffmpeg) to decode; it ships with ComfyUI. Update ComfyUI and its requirements before submitting audio.") from exc
    buffer = io.BytesIO(payload)
    try:
        with av.open(buffer) as container:
            streams = container.streams.audio
            if not streams:
                raise ValueError("OpenRouter returned audio without a decodable audio stream.")
            stream = streams[0]
            sample_rate = int(stream.rate or 0)
            if sample_rate <= 0:
                raise ValueError("OpenRouter returned audio with an invalid sample rate.")
            channels = len(stream.layout.channels)
            if channels < 1:
                raise ValueError("OpenRouter returned audio with an invalid channel layout.")
            format_name = stream.format.name if stream.format else "flt"
            planar = stream.format.is_planar if stream.format else False
            frames = []
            for frame in container.decode(stream):
                array = frame.to_ndarray()
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                elif array.ndim == 2 and not planar and array.shape[0] == 1 and channels > 1:
                    # packed (interleaved) layouts deliver shape [1, samples*channels].
                    array = array.reshape(channels, -1)
                frames.append(array)
            if not frames:
                raise ValueError("OpenRouter returned audio with no decodable samples.")
            import numpy as np
            # Planar frames stack along axis 0 (one plane per channel); packed
            # frames arrive as single interleaved planes with shape [1, samples*channels].
            if planar and channels > 1:
                data = np.concatenate(frames, axis=1)
            else:
                data = np.concatenate([f.reshape(1, -1) for f in frames], axis=1)
            if data.shape[0] != channels and data.shape[0] == 1:
                data = data.reshape(channels, -1)
            if data.shape[0] != channels:
                raise ValueError("OpenRouter returned audio whose channel planes do not match the declared layout.")
            samples = _frame_to_float(data, format_name)
            waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
            return {"waveform": waveform, "sample_rate": sample_rate}
    except (ValueError, MemoryError) as exc:
        raise
    except Exception as exc:
        raise RuntimeError(f"OpenRouter returned undecodable audio ({type(exc).__name__}).") from None