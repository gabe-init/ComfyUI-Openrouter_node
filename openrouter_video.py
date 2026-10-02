"""OpenRouter video jobs, with resumable failures and native ComfyUI output."""

import base64
import binascii
import io
import math
import re
import time
from urllib.parse import quote, urljoin, urlsplit

import requests

API_ORIGIN = "https://openrouter.ai"
API_BASE = API_ORIGIN + "/api/v1/videos"
MAX_VIDEO_BYTES = 512 * 1024 * 1024
POLL_INTERVAL = 5.0
MODES = {"text_to_video", "image_to_video", "start_end_frame_to_video", "reference_to_video"}
MODE_ALIASES = {"first_frame": "image_to_video", "first_last_frame": "start_end_frame_to_video", "reference_images": "reference_to_video"}


class VideoJobError(RuntimeError):
    """An error that retains the already-submitted job for explicit resumption."""

    def __init__(self, message, job_id=""):
        self.job_id = job_id
        self.openrouter_job_id = job_id
        if job_id:
            message += f" Video job: {job_id}. Set video_job_id to this ID to resume without submitting again."
        super().__init__(message)


def _safe_detail(value, api_key):
    text = str(value)
    if isinstance(api_key, str) and api_key:
        text = text.replace(api_key, "[redacted]")
    text = re.sub(r"data:[^\s\"']+", "[media omitted]", text)
    return text[:500]


def _job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", value):
        raise ValueError("OpenRouter returned an invalid video job ID.")
    return value


def _url(value, base=API_ORIGIN, *, require_origin=True):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("OpenRouter returned an empty video URL.")
    result = urljoin(base, value)
    parsed = urlsplit(result)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("OpenRouter returned an invalid video URL.") from exc
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Video URLs must use HTTPS without credentials or fragments.")
    if require_origin and (parsed.hostname.lower() != "openrouter.ai" or port not in (None, 443)):
        raise ValueError("Refusing to send OpenRouter credentials to another origin.")
    return result


def _native_video_support():
    try:
        from comfy_api.input_impl import VideoFromFile
        import av
    except ImportError as exc:
        raise ValueError("Video output requires ComfyUI 0.3.31 or newer with native VideoFromFile and compatible PyAV support. Update ComfyUI and its requirements before submitting a video.") from exc

    def validate_media(buffer):
        buffer.seek(0)
        try:
            with av.open(buffer, mode="r") as container:
                streams = container.streams.video
                if not streams or streams[0].width <= 0 or streams[0].height <= 0:
                    raise ValueError("Downloaded content has no valid video stream.")
                if next(container.decode(video=0), None) is None:
                    raise ValueError("Downloaded content has no decodable video frame.")
        finally:
            buffer.seek(0)

    return VideoFromFile, validate_media


def _interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def _require_model(model):
    if __package__:
        from .openrouter_catalog import require_model
    else:
        from openrouter_catalog import require_model
    return require_model("video", model)


def _supported_model(metadata):
    if __package__:
        from .openrouter_catalog import is_supported_video_model
    else:
        from openrouter_catalog import is_supported_video_model
    return is_supported_video_model(metadata)


def _payload(model, prompt, reference_urls, mode, duration, resolution, aspect_ratio, generate_audio, seed):
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"Unsupported video mode: {mode}.")
    if not isinstance(prompt, str):
        raise ValueError("The video prompt must be text.")
    if not isinstance(generate_audio, bool):
        raise ValueError("The generated-audio setting must be true or false.")
    images = list(reference_urls or [])
    expected = {"text_to_video": 0, "image_to_video": 1, "start_end_frame_to_video": 2}.get(mode)
    if expected is not None and len(images) != expected:
        raise ValueError(f"{mode} requires exactly {expected} connected image(s); received {len(images)}.")
    if mode == "reference_to_video" and not images:
        raise ValueError("reference_to_video requires at least one connected image.")
    if mode == "text_to_video" and not prompt.strip():
        raise ValueError("A prompt is required for text-to-video generation.")
    for index, image in enumerate(images):
        if not isinstance(image, str):
            raise ValueError("Video image references must be image data URLs or HTTPS URLs.")
        image = image.strip()
        if image.startswith("data:"):
            match = re.fullmatch(r"data:image/(?:png|jpeg|jpg|webp|gif);base64,(.+)", image, re.DOTALL)
            try:
                if not match or not base64.b64decode(match[1], validate=True):
                    raise ValueError
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Video image references contain an invalid image data URL.") from exc
        else:
            parsed = urlsplit(image)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("Video image references must use absolute HTTPS URLs.")
            image = _url(image, require_origin=False)
        images[index] = image

    try:
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise ValueError
        seed_value = int(seed)
        if seed_value < 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Video seed must be a nonnegative whole number.") from exc

    metadata = _require_model(model)
    if not _supported_model(dict(metadata, id=model)):
        raise ValueError("This model requires video editing, upscaling, or avatar inputs that this node does not support.")
    frames = set(metadata.get("supported_frame_images") or [])
    if mode in {"image_to_video", "start_end_frame_to_video"} and "first_frame" not in frames:
        raise ValueError(f"{model} does not advertise first-frame input support.")
    if mode == "start_end_frame_to_video" and "last_frame" not in frames:
        raise ValueError(f"{model} does not advertise last-frame input support.")
    # Reference support and limits vary by provider. Do not invent a universal
    # limit or infer support from marketing prose; honor explicit metadata.
    params = metadata.get("supported_parameters") or {}
    refs = params.get("input_references", {}) if isinstance(params, dict) else {}
    if mode == "reference_to_video" and isinstance(refs, dict):
        if refs.get("max") is not None and len(images) > refs["max"]:
            raise ValueError(f"{model} accepts at most {refs['max']} reference images.")
        if refs.get("min") is not None and len(images) < refs["min"]:
            raise ValueError(f"{model} requires at least {refs['min']} reference images.")

    payload = {"model": model}
    if prompt.strip():
        payload["prompt"] = prompt.strip()
    if duration not in (None, "", "auto", 0, "0"):
        try:
            seconds = int(duration)
            if isinstance(duration, bool) or float(duration) != seconds or seconds < 1:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Video duration must be a positive whole number of seconds or auto.") from exc
        supported = metadata.get("supported_durations") or []
        if supported and seconds not in supported:
            raise ValueError(f"{model} does not support duration {seconds}; supported values: {supported}.")
        payload["duration"] = seconds
    for field, value, catalog_field in (
        ("resolution", resolution, "supported_resolutions"),
        ("aspect_ratio", aspect_ratio, "supported_aspect_ratios"),
    ):
        if value in (None, "", "auto"):
            continue
        supported = metadata.get(catalog_field) or []
        if supported and value not in supported:
            raise ValueError(f"{model} does not support {field} {value}; supported values: {supported}.")
        payload[field] = value
    if generate_audio and metadata.get("generate_audio") is not True:
        raise ValueError(f"{model} does not advertise generated audio support.")
    if metadata.get("generate_audio") is True:
        payload["generate_audio"] = bool(generate_audio)
    if metadata.get("seed") is True:
        payload["seed"] = seed_value
    elif seed_value != 0:
        raise ValueError(f"{model} does not advertise seed support; use seed 0 to leave it unspecified.")
    if mode in {"image_to_video", "start_end_frame_to_video"}:
        payload["frame_images"] = [
            {"type": "image_url", "image_url": {"url": image}, "frame_type": frame}
            for image, frame in zip(images, ("first_frame", "last_frame"))
        ]
    elif mode == "reference_to_video":
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": image}} for image in images
        ]
    return payload


class _Runtime:
    def __init__(self, api_key, request_timeout, wait_timeout, http, clock, sleep, interrupt):
        self.api_key = api_key
        self.request_timeout = float(request_timeout)
        timeout = float(wait_timeout)
        if not math.isfinite(self.request_timeout) or self.request_timeout <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Video request and wait timeouts must be positive finite numbers.")
        self.http = http
        self.clock = clock
        self.sleep = sleep
        self.interrupt = interrupt
        self.deadline = clock() + timeout
        self.job_id = ""

    def check(self):
        self.interrupt()
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for the video. The remote job may still be running.")
        return remaining

    def wait(self, seconds):
        end = self.clock() + max(0, seconds)
        while self.clock() < end:
            remaining = self.check()
            interval = min(0.25, remaining, end - self.clock())
            if interval > 0:
                self.sleep(interval)
        self.check()

    def request(self, method, url, *, body=None, stream=False, media=False):
        url = _url(url, require_origin=not media)
        retries = 0
        redirects = 0
        while True:
            remaining = self.check()
            # Poll/download inactivity is bounded so interruption remains useful.
            connect_timeout = min(5, remaining / 2)
            read_timeout = min(self.request_timeout, remaining - connect_timeout, 15 if method == "GET" else self.request_timeout)
            parts = urlsplit(url)
            same_origin = parts.hostname == "openrouter.ai" and parts.port in (None, 443)
            headers = {"Authorization": f"Bearer {self.api_key}"} if same_origin else {}
            if body is not None:
                headers["Content-Type"] = "application/json"
            try:
                response = self.http.request(
                    method, url, headers=headers, json=body, stream=stream,
                    timeout=(connect_timeout, read_timeout), allow_redirects=False,
                )
            except Exception as exc:
                if method != "GET" or retries >= 3:
                    if method == "POST":
                        raise RuntimeError("Video submission did not return a job ID; its outcome is unknown. Check OpenRouter activity before submitting again. " + _safe_detail(exc, self.api_key)) from exc
                    raise RuntimeError("Video request failed: " + _safe_detail(exc, self.api_key)) from exc
                retries += 1
                self.wait(min(2 ** (retries - 1), 5))
                continue
            keep_open = False
            try:
                # A successful submit may already have created a billed job.
                # Deliver its ID to the caller before honoring an interruption.
                if method != "POST":
                    self.check()
                status = response.status_code
                if method == "GET" and status in {301, 302, 303, 307, 308}:
                    redirects += 1
                    if redirects > 5:
                        raise RuntimeError("Too many video download redirects.")
                    location = response.headers.get("Location", "")
                    url = _url(location, url, require_origin=not media)
                    continue
                if method == "GET" and (status == 429 or status >= 500 or media and status in {404, 409}) and retries < 3:
                    retries += 1
                    try:
                        delay = float(response.headers.get("Retry-After", 2 ** (retries - 1)))
                    except (TypeError, ValueError):
                        delay = 2 ** (retries - 1)
                    self.wait(min(max(delay, 0.25), 30))
                    continue
                if not 200 <= status < 300:
                    try:
                        detail = response.json().get("error", f"HTTP {status}")
                    except Exception:
                        detail = f"HTTP {status}"
                    if method == "POST" and (status >= 500 or 300 <= status < 400 or status == 408):
                        raise RuntimeError(f"Video submission failed ({status}); its outcome is unknown. Check OpenRouter activity before submitting again. Details: {_safe_detail(detail, self.api_key)}")
                    raise RuntimeError(f"OpenRouter video request failed ({status}): {_safe_detail(detail, self.api_key)}")
                if stream:
                    keep_open = True
                    return response
                try:
                    result = response.json()
                    if not isinstance(result, dict):
                        raise ValueError("OpenRouter returned an invalid video response.")
                except Exception as exc:
                    if method == "POST":
                        raise RuntimeError("Video submission returned an invalid response; its outcome is unknown. Check OpenRouter activity before submitting again.") from exc
                    raise
                return result
            finally:
                if not keep_open:
                    response.close()

    def download(self):
        response = self.request("GET", f"{API_BASE}/{quote(self.job_id, safe='')}/content?index=0", stream=True, media=True)
        buffer = io.BytesIO()
        try:
            content_type = response.headers.get("Content-Type", "").lower()
            if "json" in content_type or "html" in content_type:
                raise ValueError("The video download returned a document instead of video content.")
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_VIDEO_BYTES:
                raise ValueError("Video exceeds the 512 MiB download limit.")
            # read1 returns after one buffered/socket read rather than waiting
            # to fill a large chunk, allowing deadline/cancellation checks even
            # when the provider trickles the content slowly.
            read_once = getattr(getattr(response, "raw", None), "read1", None)
            chunks = iter(lambda: read_once(64 * 1024, decode_content=True), b"") if callable(read_once) else response.iter_content(chunk_size=1024)
            for chunk in chunks:
                self.check()
                if chunk:
                    if buffer.tell() + len(chunk) > MAX_VIDEO_BYTES:
                        raise ValueError("Video exceeds the 512 MiB download limit.")
                    buffer.write(chunk)
            self.check()
            if not buffer.tell():
                raise ValueError("OpenRouter returned an empty video download.")
            buffer.seek(0)
            return buffer
        finally:
            response.close()


def generate_video(api_key, model, prompt, reference_urls=None, mode="text_to_video",
                   duration="auto", resolution="auto", aspect_ratio="auto",
                   generate_audio=False, seed=0, request_timeout=120, wait_timeout=900,
                   job_id="", *, on_job=None, _http=None, _clock=None, _sleep=None,
                   _interrupt_check=None):
    """Submit once, or resume ``job_id``; return one native VIDEO and actual usage.

    ``on_job(id)`` runs immediately after a valid ID is available, before polling.
    On local cancellation the original ComfyUI BaseException is re-raised with
    ``openrouter_job_id`` attached; no remote cancellation is implied.
    """
    runtime = None
    active_id = ""
    try:
        if job_id:
            active_id = _job_id(job_id.strip())
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("An OpenRouter API key is required for video.")
        factory, validate_media = _native_video_support()
        runtime = _Runtime(api_key.strip(), request_timeout, wait_timeout,
                           _http or requests, _clock or time.monotonic,
                           _sleep or time.sleep, _interrupt_check or _interrupt)
        if job_id:
            runtime.job_id = active_id
            if on_job:
                on_job(active_id)
            polling_url = f"{API_BASE}/{quote(active_id, safe='')}"
            result = runtime.request("GET", polling_url)
        else:
            payload = _payload(model, prompt, reference_urls, mode, duration, resolution,
                               aspect_ratio, generate_audio, seed)
            result = runtime.request("POST", API_BASE, body=payload)
            try:
                active_id = _job_id(result.get("id"))
            except ValueError as exc:
                raise RuntimeError("Video submission returned no usable job ID; its outcome is unknown. Check OpenRouter activity before submitting again.") from exc
            runtime.job_id = active_id
            if on_job:
                on_job(active_id)
            polling_url = _url(result.get("polling_url") or f"{API_BASE}/{quote(active_id, safe='')}")

        while True:
            runtime.check()
            if result.get("id") not in (None, active_id):
                raise ValueError("OpenRouter returned a different video job ID while polling.")
            status = result.get("status")
            if status == "completed":
                break
            if status in {"failed", "cancelled", "expired"}:
                detail = _safe_detail(result.get("error") or status, runtime.api_key)
                raise RuntimeError(f"OpenRouter video job {status}: {detail}")
            if status not in {"pending", "in_progress"}:
                raise ValueError(f"OpenRouter returned an unknown video job status: {status!r}.")
            runtime.wait(POLL_INTERVAL)
            result = runtime.request("GET", polling_url)

        buffer = runtime.download()
        validate_media(buffer)
        runtime.check()
        video = factory(buffer)
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        cost = usage.get("cost")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost):
            cost = None
        text = f"completed | job_id={active_id}"
        if cost is not None:
            text += f" | cost=${cost:.6f}"
        return {"video": video, "text": text, "usage": usage, "cost": cost, "job_id": active_id}
    except BaseException as exc:
        if active_id:
            # Comfy's InterruptProcessingException is a BaseException, not an
            # Exception. Keep its type so ComfyUI still recognizes cancellation.
            exc.openrouter_job_id = active_id
            if not isinstance(exc, Exception):
                if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.args = (f"Video wait interrupted. Remote job {active_id} may continue; resume with video_job_id={active_id}.",)
                raise
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, VideoJobError):
            raise
        raise VideoJobError(_safe_detail(exc, api_key), active_id) from exc
