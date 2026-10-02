"""No-spend regression tests for the video job lifecycle and request contract."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "openrouter_video_test", Path(__file__).resolve().parents[1] / "openrouter_video.py"
)
video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Response:
    def __init__(self, data=None, status=200, headers=None, chunks=None):
        self.data = data
        self.status_code = status
        self.headers = headers or {}
        self.chunks = [b"test-video"] if chunks is None else chunks
        self.closed = False

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def iter_content(self, chunk_size):
        yield from self.chunks

    def close(self):
        self.closed = True


class HTTP:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response()
        return response


class ComfyInterrupt(BaseException):
    pass


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.metadata = {
            "id": "vendor/video",
            "supported_frame_images": ["first_frame", "last_frame"],
            "supported_durations": [5, 10],
            "supported_resolutions": ["720p", "1080p"],
            "supported_aspect_ratios": ["16:9", "9:16"],
            "seed": True,
            "generate_audio": True,
        }
        self.model_patch = patch.object(video, "_require_model", return_value=self.metadata)
        self.require_model = self.model_patch.start()
        self.factory = Mock(side_effect=lambda buffer: ("native-video", buffer.getvalue()))
        self.validate = Mock()
        self.native_patch = patch.object(video, "_native_video_support", return_value=(self.factory, self.validate))
        self.native_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.addCleanup(self.native_patch.stop)

    def run_video(self, http, **kwargs):
        args = dict(api_key="private-key", model="vendor/video", prompt="a cloud",
                    _http=http, _clock=self.clock, _sleep=self.clock.sleep,
                    _interrupt_check=lambda: None)
        args.update(kwargs)
        return video.generate_video(**args)

    @staticmethod
    def submitted(status="pending"):
        return Response({"id": "job-123", "polling_url": "/api/v1/videos/job-123", "status": status}, 202)

    @staticmethod
    def completed():
        return Response({"id": "job-123", "status": "completed", "usage": {"cost": 0.1, "is_byok": False}})

    def test_submit_relative_poll_and_download_native_video(self):
        responses = [self.submitted(), self.completed(), Response(headers={"Content-Type": "video/mp4"})]
        http = HTTP(*responses)
        callback = Mock()
        result = self.run_video(http, duration="5", resolution="720p", aspect_ratio="16:9", on_job=callback)
        self.assertEqual(result["job_id"], "job-123")
        self.assertEqual(result["cost"], 0.1)
        self.assertEqual(result["video"], ("native-video", b"test-video"))
        self.assertEqual([call[0] for call in http.calls], ["POST", "GET", "GET"])
        self.assertEqual(http.calls[1][1], video.API_BASE + "/job-123")
        self.assertEqual(http.calls[2][1], video.API_BASE + "/job-123/content?index=0")
        self.assertEqual(http.calls[0][2]["json"]["duration"], 5)
        self.assertFalse(http.calls[0][2]["allow_redirects"])
        callback.assert_called_once_with("job-123")
        self.validate.assert_called_once()
        self.assertTrue(all(response.closed for response in responses))

    def test_resume_bypasses_new_job_validation_and_never_posts(self):
        http = HTTP(self.completed(), Response())
        result = self.run_video(http, model="", prompt="", job_id="job-123",
                                mode="invalid", duration=-1, reference_urls=["bad"])
        self.assertEqual(result["job_id"], "job-123")
        self.require_model.assert_not_called()
        self.assertEqual([c[0] for c in http.calls], ["GET", "GET"])

    def test_native_video_dependency_checked_before_submission(self):
        http = HTTP()
        with patch.object(video, "_native_video_support", side_effect=ImportError("update ComfyUI")):
            with self.assertRaisesRegex(video.VideoJobError, "update ComfyUI"):
                self.run_video(http)
        self.assertEqual(http.calls, [])

    def test_auto_values_are_omitted_instead_of_forced_to_minimum(self):
        http = HTTP(self.submitted("completed"), Response())
        self.run_video(http)
        payload = http.calls[0][2]["json"]
        for name in ("duration", "resolution", "aspect_ratio"):
            self.assertNotIn(name, payload)

    def test_image_modes_map_existing_inputs_in_order(self):
        images = ["data:image/png;base64,Zmlyc3Q=", "data:image/png;base64,bGFzdA=="]
        for mode, count in (("image_to_video", 1), ("start_end_frame_to_video", 2), ("reference_to_video", 2)):
            with self.subTest(mode=mode):
                http = HTTP(self.submitted("completed"), Response())
                self.run_video(http, mode=mode, reference_urls=images[:count])
                payload = http.calls[0][2]["json"]
                entries = payload["input_references" if mode == "reference_to_video" else "frame_images"]
                self.assertEqual([x["image_url"]["url"] for x in entries], images[:count])
                if mode != "reference_to_video":
                    self.assertEqual([x["frame_type"] for x in entries], ["first_frame", "last_frame"][:count])

    def test_public_node_mode_aliases(self):
        for mode, count in (("first_frame", 1), ("first_last_frame", 2), ("reference_images", 3)):
            with self.subTest(mode=mode):
                http = HTTP(self.submitted("completed"), Response())
                self.run_video(http, mode=mode, reference_urls=["https://example.com/a"] * count)
                self.assertEqual(len(http.calls), 2)

    def test_preflight_rejects_invalid_inputs_without_spending(self):
        cases = [
            {"prompt": ""}, {"mode": "unknown"}, {"duration": "7"},
            {"duration": 5.1}, {"resolution": "4K"}, {"aspect_ratio": "1:1"},
            {"mode": "image_to_video", "reference_urls": []},
            {"mode": "start_end_frame_to_video", "reference_urls": ["https://example.com/image.png"]},
            {"reference_urls": ["https://example.com/image.png"]},
            {"mode": "reference_to_video", "reference_urls": []},
            {"mode": "image_to_video", "reference_urls": ["file:///private"]},
            {"mode": "image_to_video", "reference_urls": ["data:image/png;base64,bad!"]},
            {"mode": "image_to_video", "reference_urls": ["https://user:password@example.com/image"]},
            {"mode": "image_to_video", "reference_urls": ["garbage"]},
            {"mode": "image_to_video", "reference_urls": ["//example.com/image"]},
            {"seed": -1}, {"seed": 1.2}, {"seed": True},
            {"generate_audio": "false"},
            {"model": "black-forest-labs/flux-video-edit"},
        ]
        for case in cases:
            with self.subTest(case=case):
                http = HTTP()
                with self.assertRaises(video.VideoJobError):
                    self.run_video(http, **case)
                self.assertEqual(http.calls, [])

    def test_explicit_unsupported_seed_rejected_but_default_omitted(self):
        self.metadata["seed"] = False
        http = HTTP()
        with self.assertRaisesRegex(video.VideoJobError, "seed support"):
            self.run_video(http, seed=1)
        self.assertEqual(http.calls, [])
        http = HTTP(self.submitted("completed"), Response())
        self.run_video(http, seed=0)
        self.assertNotIn("seed", http.calls[0][2]["json"])

    def test_large_comfy_seed_is_not_rounded_through_float(self):
        http = HTTP(self.submitted("completed"), Response())
        self.run_video(http, seed=2**64 - 1)
        self.assertEqual(http.calls[0][2]["json"]["seed"], 2**64 - 1)

    def test_download_checks_deadline_between_partial_socket_reads(self):
        download = Response()
        def read_once(size, decode_content):
            self.clock.now += 0.4
            return b"x"
        download.raw = Mock(read1=Mock(side_effect=read_once))
        http = HTTP(self.completed(), download)
        with self.assertRaisesRegex(video.VideoJobError, "Timed out") as caught:
            self.run_video(http, job_id="job-123", wait_timeout=1)
        self.assertEqual(download.raw.read1.call_count, 3)
        self.assertTrue(download.closed)
        self.assertEqual(caught.exception.job_id, "job-123")

    def test_advertised_frame_audio_reference_limits_enforced(self):
        self.metadata["supported_frame_images"] = ["first_frame"]
        self.metadata["generate_audio"] = False
        self.metadata["supported_parameters"] = {"input_references": {"type": "range", "min": 1, "max": 1}}
        for args in [
            {"generate_audio": True},
            {"mode": "start_end_frame_to_video", "reference_urls": ["https://example.com/a", "https://example.com/b"]},
            {"mode": "reference_to_video", "reference_urls": ["https://example.com/a", "https://example.com/b"]},
        ]:
            with self.subTest(args=args):
                http = HTTP()
                with self.assertRaises(video.VideoJobError):
                    self.run_video(http, **args)
                self.assertEqual(http.calls, [])

    def test_unknown_reference_limit_is_not_guessed(self):
        http = HTTP(self.submitted("completed"), Response())
        self.run_video(http, mode="reference_to_video", reference_urls=["https://example.com/a"] * 5)
        self.assertEqual(len(http.calls[0][2]["json"]["input_references"]), 5)

    def test_remote_polling_url_rejected_with_recoverable_job(self):
        http = HTTP(Response({"id": "job-123", "status": "pending", "polling_url": "https://attacker.example/poll"}, 202))
        with self.assertRaisesRegex(video.VideoJobError, "another origin") as caught:
            self.run_video(http)
        self.assertEqual(caught.exception.job_id, "job-123")
        self.assertEqual(len(http.calls), 1)

    def test_unsigned_content_redirect_never_forwards_key_to_cdn(self):
        http = HTTP(self.completed(), Response(status=302, headers={"Location": "https://cdn.example/video.mp4"}), Response())
        self.run_video(http, job_id="job-123")
        self.assertIn("Authorization", http.calls[1][2]["headers"])
        self.assertNotIn("Authorization", http.calls[2][2]["headers"])

    def test_poll_redirect_cannot_leak_key(self):
        redirect = Response(status=302, headers={"Location": "https://attacker.example/poll"})
        http = HTTP(redirect)
        with self.assertRaises(video.VideoJobError) as caught:
            self.run_video(http, job_id="job-123")
        self.assertEqual(len(http.calls), 1)
        self.assertTrue(redirect.closed)
        self.assertEqual(caught.exception.job_id, "job-123")

    def test_http_download_redirect_rejected(self):
        http = HTTP(self.completed(), Response(status=302, headers={"Location": "http://cdn.example/video.mp4"}))
        with self.assertRaisesRegex(video.VideoJobError, "HTTPS"):
            self.run_video(http, job_id="job-123")
        self.assertEqual(len(http.calls), 2)

    def test_terminal_states_stop_without_download_or_resubmit(self):
        for status in ("failed", "cancelled", "expired", "nonsense"):
            with self.subTest(status=status):
                http = HTTP(Response({"status": status, "error": "provider message"}))
                with self.assertRaises(video.VideoJobError) as caught:
                    self.run_video(http, job_id="job-123")
                self.assertEqual(caught.exception.job_id, "job-123")
                self.assertEqual(len(http.calls), 1)

    def test_submission_timeout_is_not_retried(self):
        http = HTTP(TimeoutError("read timed out"))
        with self.assertRaisesRegex(video.VideoJobError, "outcome is unknown"):
            self.run_video(http)
        self.assertEqual(len(http.calls), 1)

    def test_ambiguous_submit_server_or_decode_errors_never_retry(self):
        for response in [Response({"error": "upstream timeout"}, 502),
                         Response(ValueError("bad JSON"), 202), Response([], 202),
                         Response(status=307, headers={"Location": video.API_BASE})]:
            with self.subTest(status=response.status_code):
                http = HTTP(response)
                with self.assertRaisesRegex(video.VideoJobError, "outcome is unknown"):
                    self.run_video(http)
                self.assertEqual(len(http.calls), 1)

    def test_poll_response_must_belong_to_same_job(self):
        http = HTTP(Response({"id": "job-wrong", "status": "completed"}))
        with self.assertRaisesRegex(video.VideoJobError, "different video job ID") as caught:
            self.run_video(http, job_id="job-123")
        self.assertEqual(caught.exception.job_id, "job-123")
        self.assertEqual(len(http.calls), 1)

    def test_poll_transient_retry_retains_same_job(self):
        http = HTTP(Response(status=429, headers={"Retry-After": "1"}), TimeoutError("temporary"), self.completed(), Response())
        self.run_video(http, job_id="job-123")
        self.assertEqual([c[0] for c in http.calls], ["GET"] * 4)
        self.assertEqual(len({c[1] for c in http.calls[:3]}), 1)

    def test_deadline_in_poll_wait_retains_job(self):
        http = HTTP(self.submitted())
        with self.assertRaisesRegex(video.VideoJobError, "Timed out") as caught:
            self.run_video(http, wait_timeout=1)
        self.assertEqual(caught.exception.job_id, "job-123")
        self.assertLessEqual(self.clock.now, 1)
        self.assertEqual(len(http.calls), 1)

    def test_interruption_preserves_comfy_baseexception_and_submitted_id(self):
        submitted = False
        def submit():
            nonlocal submitted
            submitted = True
            return self.submitted()
        def interrupt():
            if submitted:
                raise ComfyInterrupt()
        http = HTTP(submit)
        callback = Mock()
        with self.assertRaises(ComfyInterrupt) as caught:
            self.run_video(http, on_job=callback, _interrupt_check=interrupt)
        self.assertEqual(caught.exception.openrouter_job_id, "job-123")
        callback.assert_called_once_with("job-123")
        self.assertEqual(len(http.calls), 1)

    def test_media_validation_failure_is_recoverable_without_false_success(self):
        http = HTTP(self.completed(), Response())
        self.validate.side_effect = ValueError("no video stream")
        with self.assertRaisesRegex(video.VideoJobError, "no video stream") as caught:
            self.run_video(http, job_id="job-123")
        self.assertEqual(caught.exception.job_id, "job-123")
        self.factory.assert_not_called()

    def test_reject_empty_document_and_oversized_downloads(self):
        downloads = [Response(chunks=[]), Response(headers={"Content-Type": "application/json"}),
                     Response(headers={"Content-Length": str(video.MAX_VIDEO_BYTES + 1)})]
        for download in downloads:
            with self.subTest(headers=download.headers):
                http = HTTP(self.completed(), download)
                with self.assertRaises(video.VideoJobError):
                    self.run_video(http, job_id="job-123")
                self.assertTrue(download.closed)

    def test_diagnostics_redact_secrets_and_image_data(self):
        http = HTTP(Response({"error": "private-key data:image/png;base64,secretbytes"}, 400))
        with self.assertRaises(video.VideoJobError) as caught:
            self.run_video(http)
        text = str(caught.exception)
        self.assertNotIn("private-key", text)
        self.assertNotIn("secretbytes", text)

    def test_missing_submission_id_warns_of_ambiguous_outcome(self):
        http = HTTP(Response({"status": "pending"}, 202))
        with self.assertRaisesRegex(video.VideoJobError, "outcome is unknown"):
            self.run_video(http)
        self.assertEqual(len(http.calls), 1)

    def test_request_timeout_never_exceeds_remaining_budget(self):
        http = HTTP(self.completed(), Response())
        self.run_video(http, job_id="job-123", request_timeout=120, wait_timeout=2)
        for _, _, kwargs in http.calls:
            self.assertTrue(all(t <= 2 for t in kwargs["timeout"]))
            self.assertLessEqual(sum(kwargs["timeout"]), 2)


if __name__ == "__main__":
    unittest.main()
