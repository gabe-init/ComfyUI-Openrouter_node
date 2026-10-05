"""Validate audio inputs and build OpenRouter's base64 input_audio block.

OpenRouter audio format support is provider-specific:
https://openrouter.ai/docs/guides/overview/multimodal/audio
"""

import base64
import hashlib
import io
import json
from numbers import Integral
import wave

import numpy as np
import torch


SUPPORTED_AUDIO_FORMATS = frozenset(
    {"wav", "mp3", "aiff", "aac", "ogg", "flac", "m4a", "pcm16", "pcm24"}
)
_EXTENSION_FORMATS = {name: name for name in SUPPORTED_AUDIO_FORMATS}
_EXTENSION_FORMATS["aif"] = "aiff"

# PCM16 WAV above this size (base64 inflates it ~33%) risks a 413 from OpenRouter's
# gateway; MP3-compress long clips instead of sending raw, uncompressed PCM.
_MAX_WAV_BYTES = 15 * 1024 * 1024
_MP3_BIT_RATE = 128000


def _detected_format(data):
    """Recognize common container signatures without pretending to decode them."""
    # ID3 tags can precede both MP3 and AAC. Inspect the following audio header
    # rather than treating a tag alone as proof of a particular codec.
    if data.startswith(b"ID3"):
        if len(data) < 10 or any(byte & 0x80 for byte in data[6:10]):
            return None
        tag_size = sum(byte << shift for byte, shift in zip(data[6:10], (21, 14, 7, 0)))
        data = data[10 + tag_size:]
    if len(data) >= 12:
        if data[:4] in (b"RIFF", b"RIFX", b"RF64") and data[8:12] == b"WAVE":
            return "wav"
        if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
            return "aiff"
        if data[4:8] == b"ftyp":
            return "m4a"
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"OggS"):
        return "ogg"
    if data.startswith(b"ADIF"):
        return "aac"
    if len(data) >= 2 and data[0] == 0xFF:
        if data[1] & 0xF6 == 0xF0:
            return "aac"  # ADTS, MPEG-2 or MPEG-4
        if data[1] & 0xE6 == 0xE2 and data[1] & 0x18 != 0x08:
            return "mp3"  # MPEG audio frame with Layer III and a valid version
    return None


def _raw_audio(audio):
    data = audio.get("bytes")
    if not isinstance(data, bytes) or not data:
        raise ValueError("Audio 'bytes' must contain nonempty file bytes.")

    filename = audio.get("filename")
    extension_format = None
    if filename is not None:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Audio 'filename' must be a nonempty string when provided.")
        filename = filename.strip()
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        extension_format = _EXTENSION_FORMATS.get(extension)

    explicit_format = audio.get("format")
    if explicit_format is not None:
        if not isinstance(explicit_format, str) or not explicit_format.strip():
            raise ValueError("Audio 'format' must be a nonempty supported format name.")
        audio_format = explicit_format.strip().lower()
        if audio_format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"Unsupported audio format: {audio_format}.")
        if extension_format is not None and extension_format != audio_format:
            raise ValueError("Audio 'format' conflicts with the filename extension.")
    else:
        audio_format = extension_format
        if audio_format is None:
            raise ValueError("Cannot determine audio format; use a supported extension or explicit 'format'.")

    if audio_format in ("pcm16", "pcm24"):
        # Raw PCM has no signature: any byte prefix can be legitimate samples.
        sample_width = 2 if audio_format == "pcm16" else 3
        if len(data) % sample_width:
            raise ValueError(f"Audio {audio_format} bytes must contain complete samples.")
    else:
        detected_format = _detected_format(data)
        if detected_format is not None and detected_format != audio_format:
            raise ValueError(
                f"Audio content appears to be {detected_format}, but its format is {audio_format}."
            )
    return data, audio_format, {"kind": "file", "format": audio_format}


def _mp3_encode(waveform, channels, sample_rate):
    """Compress a [channels, samples] float32 waveform to MP3 bytes via PyAV."""
    import av  # ComfyUI ships PyAV; import lazily, only needed for large clips.

    layout = "mono" if channels == 1 else "stereo"
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp3")
    stream = container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
    stream.bit_rate = _MP3_BIT_RATE  # type: ignore[reportAttributeAccessIssue]
    frame = av.AudioFrame.from_ndarray(
        waveform.transpose(0, 1).reshape(1, -1).numpy(), format="flt", layout=layout,
    )
    frame.sample_rate = sample_rate
    frame.pts = 0
    container.mux(stream.encode(frame))  # type: ignore[reportAttributeAccessIssue]
    container.mux(stream.encode(None))  # type: ignore[reportAttributeAccessIssue]
    container.close()
    return buffer.getvalue()


def _native_audio(audio, encoding="auto"):
    if encoding not in ("auto", "wav", "mp3"):
        raise ValueError("Audio encoding must be 'auto', 'wav', or 'mp3'.")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or not waveform.is_floating_point():
        raise ValueError("Native AUDIO waveform must be a floating-point torch.Tensor.")
    if waveform.ndim != 3:
        raise ValueError("Native AUDIO waveform must have shape [1, channels, samples].")
    batch, channels, samples = waveform.shape
    if batch != 1:
        raise ValueError("Native AUDIO supports exactly one batch item; select one clip first.")
    if channels not in (1, 2):
        raise ValueError("Native AUDIO supports mono or stereo; downmix other channel layouts first.")
    if samples == 0:
        raise ValueError("Native AUDIO waveform must contain at least one sample.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0:
        raise ValueError("Native AUDIO sample_rate must be a positive integer.")
    sample_rate = int(sample_rate)
    # WAV stores its byte rate in an unsigned 32-bit field.
    if sample_rate * channels * 2 > 0xFFFFFFFF:
        raise ValueError("Native AUDIO sample_rate is too large for PCM16 WAV.")

    waveform = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("Native AUDIO waveform must contain only finite samples.")
    if bool((waveform.abs() > 1).any()):
        # Lossy codecs (MP3/AAC) can decode with slight overshoot above 1.0;
        # PCM16 quantization clamps anyway, so warn and clamp instead of failing.
        print("Warning: Native AUDIO samples outside [-1, 1] detected; clamping to the valid range.")
        waveform = waveform.clamp_(-1.0, 1.0)

    if encoding == "mp3":
        # Forced MP3: skip building a WAV buffer that would only be discarded.
        try:
            mp3_bytes = _mp3_encode(waveform[0], channels, sample_rate)
        except Exception as error:
            raise ValueError(
                "audio_encoding is set to 'mp3' but MP3 compression is unavailable "
                "(PyAV/libmp3lame missing?). Switch audio_encoding to 'auto' or 'wav'."
            ) from error
        return mp3_bytes, "mp3", {
            "kind": "native_mp3",
            "format": "mp3",
            "shape": [batch, channels, samples],
            "sample_rate": sample_rate,
        }

    # Preserve channels and duration; PCM is interleaved by sample, not channel.
    samples_array = waveform[0].transpose(0, 1).numpy()
    pcm = np.rint(np.where(samples_array < 0, samples_array * 32768, samples_array * 32767))
    pcm_bytes = pcm.astype("<i2").tobytes(order="C")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm_bytes)
    wav_bytes = buffer.getvalue()

    if encoding == "auto" and len(wav_bytes) > _MAX_WAV_BYTES:
        try:
            mp3_bytes = _mp3_encode(waveform[0], channels, sample_rate)
        except Exception as error:
            raise ValueError(
                f"Native AUDIO clip is too large to send as WAV ({len(wav_bytes)} bytes) "
                "and MP3 compression is unavailable. Trim the clip or install PyAV/ffmpeg."
            ) from error
        return mp3_bytes, "mp3", {
            "kind": "native_mp3",
            "format": "mp3",
            "shape": [batch, channels, samples],
            "sample_rate": sample_rate,
        }

    return wav_bytes, "wav", {
        "kind": "native",
        "format": "wav",
        "shape": [batch, channels, samples],
        "sample_rate": sample_rate,
    }


def _normalize_audio(audio_data, encoding="auto"):
    if not isinstance(audio_data, dict):
        raise ValueError("Audio input must be a native AUDIO dictionary or a dictionary of file bytes.")
    if "waveform" in audio_data or "sample_rate" in audio_data:
        if "bytes" in audio_data:
            raise ValueError("Audio input must not mix native waveform and raw file bytes.")
        return _native_audio(audio_data, encoding)
    return _raw_audio(audio_data)  # raw files pass through unmodified regardless of encoding


def _fingerprint(data, metadata):
    hasher = hashlib.sha256()
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(data)
    return hasher.hexdigest()


def audio_fingerprint(audio_data, encoding="auto"):
    """Return the validated payload identity, without allocating base64 text.

    Invalid audio raises ValueError rather than producing a reusable error key.
    """
    data, _audio_format, metadata = _normalize_audio(audio_data, encoding)
    return _fingerprint(data, metadata)


def prepare_audio(audio_data, encoding="auto"):
    """Return an input_audio content block and its deterministic fingerprint.

    encoding: "auto" (WAV, MP3 only above ~15 MB), "wav" (always WAV), or
    "mp3" (always MP3, skips building a WAV that would be discarded). Raw
    file bytes always pass through unmodified regardless of this setting.
    """
    data, audio_format, metadata = _normalize_audio(audio_data, encoding)
    return {
        "block": {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(data).decode("ascii"),
                "format": audio_format,
            },
        },
        "fingerprint": _fingerprint(data, metadata),
    }
