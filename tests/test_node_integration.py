import sys
import types
import unittest
import inspect
import io
from unittest.mock import Mock, patch

from .test_request_timeout import load_node_module


class NodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_node_module()
        self.package = self.module.__package__
        catalog = types.ModuleType(f"{self.package}.openrouter_catalog")
        catalog.get_model = Mock(return_value={})
        catalog.get_catalog = Mock(return_value={"chat": [], "image": [], "video": []})
        catalog.require_model = Mock(return_value={"supported_parameters": {}, "architecture": {
            "input_modalities": ["text", "image", "audio"], "output_modalities": ["text"],
        }})
        self.catalog = catalog
        catalog_patch = patch.dict(sys.modules, {catalog.__name__: catalog})
        catalog_patch.start()
        self.addCleanup(catalog_patch.stop)
        self.node = self.module.OpenRouterNode()
        self.node.fetch_credits = Mock(return_value="Remaining: $1.000")
        self.node.count_tokens = Mock(return_value=1)
        self.args = dict(api_key="test-key", system_prompt="system", user_message_box="hello",
                         model="example/model", web_search=False, cheapest=False, fastest=False,
                         temperature=1, pdf_engine="auto", chat_mode=False)
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "done"}}],
                                      "usage": {"cost": .01}, "service_tier": "flex"}
        self.module.requests.post.return_value = response

    def test_existing_output_positions_and_new_optional_inputs(self):
        self.assertEqual(self.node.RETURN_TYPES, ("STRING", "IMAGE", "STRING", "STRING", "VIDEO", "AUDIO"))
        self.assertEqual(self.node.RETURN_NAMES, ("Output", "image", "Stats", "Credits", "video", "audio"))
        schema = self.node.INPUT_TYPES()
        self.assertEqual(list(schema["required"]), ["api_key", "system_prompt", "user_message_box", "model",
            "web_search", "cheapest", "fastest", "aspect_ratio", "image_resolution", "reasoning_effort",
            "seed", "temperature", "pdf_engine", "chat_mode", "request_timeout"])
        self.assertEqual(schema["optional"]["request_type"][1]["default"], "chat")
        self.assertEqual(schema["optional"]["request_type"][0], ["chat", "image", "video", "audio"])
        self.assertIn("tts_voice", schema["optional"])
        # Reference inputs must exist in the schema: sockets are instantiated
        # from it at node creation, and missing declarations mean missing slots.
        for name, kind in (("tts_reference_audio", "AUDIO"), ("tts_reference_image", "IMAGE"),
                           ("tts_reference_text", "STRING")):
            self.assertIn(name, schema["optional"], f"{name} must be declared as an input")
            self.assertEqual(schema["optional"][name][0], kind)
        self.assertEqual(schema["hidden"], {"unique_id": "UNIQUE_ID"})

    def test_explicit_tier_does_not_replace_cheapest_routing(self):
        result = self.node.generate_response(**{**self.args, "cheapest": True}, service_tier="flex")
        payload = self.module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "example/model:floor")
        self.assertEqual(payload["service_tier"], "flex")
        self.assertIn("Service tier: flex", result[2])
        self.assertEqual(len(result), 5)

    def test_auto_tier_is_omitted(self):
        self.node.generate_response(**self.args)
        self.assertNotIn("service_tier", self.module.requests.post.call_args.kwargs["json"])

    def test_connected_empty_prompt_overrides_internal_prompt(self):
        self.node.generate_response(**self.args, user_message_input="")
        self.assertEqual(self.module.requests.post.call_args.kwargs["json"]["messages"][-1]["content"], "")

    def test_invalid_audio_and_pdf_never_submit(self):
        audio = types.ModuleType(f"{self.package}.openrouter_audio")
        audio.prepare_audio = Mock(side_effect=ValueError("invalid audio"))
        with patch.dict(sys.modules, {audio.__name__: audio}):
            result = self.node.generate_response(**self.args, audio_data={})
        self.assertIn("invalid audio", result[0])
        self.module.requests.post.assert_not_called()
        self.node.generate_response(**self.args, pdf_data={})
        self.module.requests.post.assert_not_called()

    def test_audio_attached_to_chat_content(self):
        audio = types.ModuleType(f"{self.package}.openrouter_audio")
        audio.prepare_audio = Mock(return_value={"block": {"type": "input_audio", "input_audio": {"data": "test", "format": "wav"}}})
        with patch.dict(sys.modules, {audio.__name__: audio}):
            self.node.generate_response(**self.args, audio_data={"bytes": b"test"})
        content = self.module.requests.post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertEqual(content[1]["type"], "input_audio")

    def test_audio_rejected_before_post_for_known_text_only_model(self):
        self.catalog.require_model.return_value = {"architecture": {"input_modalities": ["text", "image"]}}
        result = self.node.generate_response(**self.args, audio_data={"bytes": b"test"})
        self.assertIn("does not support audio", result[0])
        self.module.requests.post.assert_not_called()

    def test_chat_waits_for_capabilities_once_before_image_payload(self):
        self.catalog.get_model.return_value = {"supported_parameters": {
            "resolution": {"type": "enum", "values": ["1K", "2K"]},
            "aspect_ratio": {"type": "enum", "values": ["16:9"]},
        }}
        self.catalog.require_model.return_value = {"id": "google/gemini-3-pro-image", "architecture": {
            "input_modalities": ["text", "image"], "output_modalities": ["text", "image"],
        }}
        self.node.generate_response(**self.args, aspect_ratio="16:9", image_resolution="2K")
        self.catalog.require_model.assert_called_once_with("chat", "example/model")
        self.catalog.get_model.assert_called_once_with("image", "google/gemini-3-pro-image")
        payload = self.module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["modalities"], ["text", "image"])
        self.assertEqual(payload["image_config"], {"aspect_ratio": "16:9", "image_size": "2K"})

    def test_gpt_chat_images_omit_inherited_gemini_resolution(self):
        self.catalog.require_model.return_value = {"id": "openai/gpt-5.4-image-2", "architecture": {
            "output_modalities": ["text", "image"],
        }}
        self.catalog.get_model.return_value = {"supported_parameters": {
            "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]},
        }}
        self.node.generate_response(**self.args, image_resolution="1K", aspect_ratio="16:9")
        payload = self.module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["image_config"], {"aspect_ratio": "16:9"})
        self.assertNotIn("image_size", payload["image_config"])

    def test_gpt_chat_images_reject_explicit_unsupported_settings_before_post(self):
        self.catalog.require_model.return_value = {"id": "openai/gpt-5.4-image-2", "architecture": {
            "output_modalities": ["text", "image"],
        }}
        self.catalog.get_model.return_value = {"supported_parameters": {
            "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]},
        }}
        for controls in ({"image_resolution": "2K"}, {"aspect_ratio": "1:8"}):
            with self.subTest(controls=controls):
                result = self.node.generate_response(**self.args, **controls)
                self.assertIn("Error:", result[0])
        self.module.requests.post.assert_not_called()

    def test_gemini_chat_images_preserve_1k_and_normalize_old_half_k_value(self):
        self.catalog.require_model.return_value = {"id": "google/gemini-3.1-flash-image-preview", "architecture": {
            "output_modalities": ["text", "image"],
        }}
        self.catalog.get_model.return_value = {"supported_parameters": {
            "resolution": {"type": "enum", "values": ["512", "1K", "2K", "4K"]},
        }}
        for value, expected in (("1K", "1K"), ("0.5K", "512"), ("512", "512")):
            with self.subTest(resolution=value):
                self.node.generate_response(**self.args, image_resolution=value)
                payload = self.module.requests.post.call_args.kwargs["json"]
                self.assertEqual(payload["image_config"], {"image_size": expected})

    def test_unknown_chat_image_counterpart_allows_defaults_only(self):
        self.catalog.require_model.return_value = {"id": "openrouter/auto", "architecture": {
            "output_modalities": ["text", "image"],
        }}
        self.catalog.get_model.side_effect = ValueError("no image counterpart")
        self.node.generate_response(**self.args)
        self.assertNotIn("image_config", self.module.requests.post.call_args.kwargs["json"])
        self.module.requests.post.reset_mock()
        result = self.node.generate_response(**self.args, aspect_ratio="16:9")
        self.assertIn("Cannot verify", result[0])
        self.module.requests.post.assert_not_called()

    def test_chat_catalog_lookup_preserves_free_variant_before_routing_suffixes(self):
        self.catalog.require_model.side_effect = ValueError("routing suffix not catalog ID")
        self.catalog.get_model.return_value = {"architecture": {
            "input_modalities": ["text"], "output_modalities": ["text"],
        }}
        self.node.generate_response(**{**self.args, "model": "vendor/model:free:floor:online"})
        self.catalog.require_model.assert_called_once_with("chat", "vendor/model:free:floor:online")
        self.catalog.get_model.assert_called_once_with("chat", "vendor/model:free")
        self.module.requests.post.assert_called_once()

    def test_exact_catalog_variant_is_not_stripped(self):
        self.node.generate_response(**{**self.args, "model": "vendor/model:free"})
        self.catalog.require_model.assert_called_once_with("chat", "vendor/model:free")
        self.catalog.get_model.assert_not_called()

    def test_unavailable_media_metadata_fails_before_submission(self):
        self.catalog.require_model.side_effect = ValueError("discovery unavailable")
        for kwargs in ({"audio_data": {}}, {"image_1": object()}, {"aspect_ratio": "16:9"}, {"image_resolution": "2K"}):
            with self.subTest(inputs=list(kwargs)):
                result = self.node.generate_response(**self.args, **kwargs)
                self.assertIn("Cannot verify", result[0])
        self.module.requests.post.assert_not_called()

    def test_plain_custom_chat_fallback_waits_for_lookup(self):
        self.catalog.require_model.side_effect = ValueError("custom ID not in public catalog")
        result = self.node.generate_response(**self.args)
        self.catalog.require_model.assert_called_once_with("chat", "example/model")
        self.module.requests.post.assert_called_once()
        self.assertEqual(result[0], "done")

    def test_custom_validation_does_not_disable_static_enum_and_number_checks(self):
        parameters = inspect.signature(self.node.VALIDATE_INPUTS).parameters
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
        for name in ("seed", "request_type", "service_tier", "video_mode", "request_timeout"):
            self.assertNotIn(name, parameters)

    def test_tts_voice_bypasses_static_combo_validation(self):
        # The widget ships with only ["auto"]; the frontend fills voices from
        # the speech catalog after load. A saved "george" must not be rejected
        # against the stale snapshot - the live check runs at execution time.
        self.assertIn("tts_voice", inspect.signature(self.node.VALIDATE_INPUTS).parameters)

    def test_null_chat_content_returns_a_string(self):
        self.module.requests.post.return_value.json.return_value = {
            "choices": [{"message": {"content": None}}], "usage": {},
        }
        result = self.node.generate_response(**self.args)
        self.assertEqual(result[0], "")

    def test_video_resume_bypasses_unrelated_inputs(self):
        video = types.ModuleType(f"{self.package}.openrouter_video")
        video.generate_video = Mock(return_value={"video": "native-video", "job_id": "job-123", "text": "done"})
        with patch.dict(sys.modules, {video.__name__: video}):
            result = self.node.generate_response(**self.args, request_type="video", video_job_id="job-123",
                                                 pdf_data={}, audio_data={}, image_1=object())
        self.assertEqual(result[4], "native-video")
        self.assertEqual(video.generate_video.call_args.kwargs["reference_urls"], [])
        self.module.requests.post.assert_not_called()

    def test_video_resume_cache_bypasses_audio_validation_and_reruns_polling(self):
        audio = types.ModuleType(f"{self.package}.openrouter_audio")
        audio.audio_fingerprint = Mock(side_effect=ValueError("stale audio"))
        with patch.dict(sys.modules, {audio.__name__: audio}):
            first = self.node.IS_CHANGED(**self.args, request_type="video", video_job_id="job-123", audio_data={})
            second = self.node.IS_CHANGED(**self.args, request_type="video", video_job_id="job-123", audio_data={})
        audio.audio_fingerprint.assert_not_called()
        self.assertNotEqual(first, second, "resuming a pending job must not reuse a cached timeout result")

    def test_new_video_validation_error_does_not_offer_an_old_job(self):
        self.node.last_video_job_id = "previous-job"
        with self.assertRaises(RuntimeError) as caught:
            self.node.generate_response(**self.args, request_type="video", audio_data={})
        self.assertNotIn("previous-job", str(caught.exception))
        self.assertNotIn("resume with", str(caught.exception))
        self.module.requests.post.assert_not_called()

    def test_video_failure_stops_graph_with_visible_resume_id(self):
        video = types.ModuleType(f"{self.package}.openrouter_video")
        def generate(*args, **kwargs):
            kwargs["on_job"]("job-123")
            raise ValueError("test-key data:image/png;base64,AAAA download timed out")
        video.generate_video = generate
        with patch.dict(sys.modules, {video.__name__: video}):
            with self.assertRaisesRegex(RuntimeError, "video_job_id=job-123") as caught:
                self.node.generate_response(**self.args, request_type="video")
        self.assertEqual(caught.exception.openrouter_job_id, "job-123")
        self.assertNotIn("test-key", str(caught.exception))
        self.assertNotIn("AAAA", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        self.node.fetch_credits.assert_not_called()

    def test_video_job_event_precedes_native_interruption(self):
        class ComfyInterrupt(BaseException):
            pass
        interrupt = ComfyInterrupt()
        video = types.ModuleType(f"{self.package}.openrouter_video")
        server = types.ModuleType("server")
        instance = types.SimpleNamespace(client_id="client-1", send_sync=Mock())
        server.PromptServer = types.SimpleNamespace(instance=instance)
        def generate(*args, **kwargs):
            kwargs["on_job"]("job-123")
            instance.send_sync.assert_called_once_with(
                "openrouter.video_job", {"node_id": "12:34", "job_id": "job-123"}, sid="client-1")
            raise interrupt
        video.generate_video = generate
        with patch.dict(sys.modules, {video.__name__: video, "server": server}):
            with self.assertRaises(ComfyInterrupt) as caught:
                self.node.generate_response(**self.args, request_type="video", unique_id="12:34")
        self.assertIs(caught.exception, interrupt)
        self.assertEqual(self.node.last_video_job_id, "job-123")
        self.node.fetch_credits.assert_not_called()

    def test_failed_video_event_delivery_keeps_job_and_logs_recovery(self):
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
            client_id="client-1", send_sync=Mock(side_effect=RuntimeError("test-key"))))
        with patch.dict(sys.modules, {"server": server}), self.assertLogs(self.module.__name__, level="WARNING") as logs:
            self.node._remember_video_job("job-123", "12")
        self.assertEqual(self.node.last_video_job_id, "job-123")
        self.assertIn("video_job_id=job-123", " ".join(logs.output))
        self.assertNotIn("test-key", " ".join(logs.output))

    def test_audio_cache_changes_when_fingerprint_changes_and_contains_no_key(self):
        audio = types.ModuleType(f"{self.package}.openrouter_audio")
        audio.audio_fingerprint = Mock(side_effect=["first", "second"])
        with patch.dict(sys.modules, {audio.__name__: audio}):
            first = self.node.IS_CHANGED(**self.args, audio_data={"value": 1})
            second = self.node.IS_CHANGED(**self.args, audio_data={"value": 2})
        self.assertNotEqual(first, second)
        self.assertNotIn("test-key", repr(first))

    def test_chat_ignores_hidden_video_and_native_image_controls(self):
        self.node.generate_response(**self.args, video_duration="15", video_mode="reference_images",
                                    image_quality="high", image_background="transparent")
        payload = self.module.requests.post.call_args.kwargs["json"]
        for field in ("duration", "video_mode", "quality", "background"):
            self.assertNotIn(field, payload)

    def test_native_image_output_preserves_transparent_pixels(self):
        import numpy as np
        import torch
        from PIL import Image

        encoded = io.BytesIO()
        Image.new("RGBA", (2, 2), (10, 20, 30, 40)).save(encoded, format="PNG")
        images = types.ModuleType(f"{self.package}.openrouter_images")
        images.generate_images = Mock(return_value={"images": [encoded.getvalue()], "usage": {}})
        with patch.dict(sys.modules, {images.__name__: images}), patch.multiple(
            self.module, torch=torch, np=np, Image=Image,
        ):
            result = self.node.generate_response(**self.args, request_type="image", image_background="transparent")
        self.assertEqual(tuple(result[1].shape), (1, 2, 2, 4))
        self.assertAlmostEqual(result[1][0, 0, 0, 3].item(), 40 / 255, places=6)
        self.module.requests.post.assert_not_called()

    def test_secret_and_media_redaction(self):
        text = self.node._safe_error(ValueError('test-key data:image/png;base64,AAAA'), "test-key")
        self.assertNotIn("test-key", text)
        self.assertNotIn("AAAA", text)


class TtsIntegrationTests(unittest.TestCase):
    """The audio request_type routes through openrouter_tts, not chat/completions."""

    def setUp(self):
        self.module = load_node_module()
        self.package = self.module.__package__
        catalog = types.ModuleType(f"{self.package}.openrouter_catalog")
        catalog.get_model = Mock(return_value={})
        catalog.get_catalog = Mock(return_value={"chat": [], "image": [], "video": [], "speech": []})
        catalog.require_model = Mock(return_value={"supported_parameters": {}, "architecture": {
            "output_modalities": ["speech"], "input_modalities": ["text"],
        }, "supported_voices": ["george"]})
        catalog_patch = patch.dict(sys.modules, {catalog.__name__: catalog})
        catalog_patch.start()
        self.addCleanup(catalog_patch.stop)
        self.node = self.module.OpenRouterNode()
        self.node.fetch_credits = Mock(return_value="Remaining: $1.000")
        self.node.count_tokens = Mock(return_value=1)
        self.args = dict(api_key="test-key", system_prompt="system", user_message_box="hello",
                         model="example/tts", web_search=False, cheapest=False, fastest=False,
                         temperature=1, pdf_engine="auto", chat_mode=False, request_type="audio")
        import torch as real_torch
        waveform = real_torch.zeros((1, 1, 4))
        self.audio_dict = {"waveform": waveform, "sample_rate": 24000}

    def tts_module(self, result=None, side_effect=None):
        module = types.ModuleType(f"{self.package}.openrouter_tts")
        module.generate_speech = Mock(return_value=result or (self.audio_dict, "gen-1"))
        if side_effect:
            module.generate_speech.side_effect = side_effect
        return module

    def test_audio_mode_calls_tts_and_returns_audio_output(self):
        tts = self.tts_module()
        with patch.dict(sys.modules, {tts.__name__: tts}):
            result = self.node.generate_response(**self.args, tts_voice="george", tts_format="mp3")
        self.assertEqual(tts.generate_speech.call_args.args[:3], ("test-key", "example/tts", "hello"))
        kwargs = tts.generate_speech.call_args.kwargs
        self.assertEqual(kwargs["voice"], "george")
        self.assertEqual(kwargs["response_format"], "mp3")
        self.assertEqual(result[0], "audio | model=example/tts | generation=gen-1")
        self.assertIs(result[5], self.audio_dict)
        self.module.requests.post.assert_not_called()

    def test_audio_mode_parses_speaker_turns_json(self):
        tts = self.tts_module()
        turns = '{"text": "Hi Jane", "voice": "Kore"}\n\n{"text": "Hey"}'
        with patch.dict(sys.modules, {tts.__name__: tts}):
            self.node.generate_response(**self.args, tts_speakers=turns)
        self.assertEqual(tts.generate_speech.call_args.kwargs["speaker_turns"],
                         [{"text": "Hi Jane", "voice": "Kore"}, {"text": "Hey"}])

    def test_audio_error_stops_graph_instead_of_returning_none_audio(self):
        tts = self.tts_module(side_effect=RuntimeError("declined"))
        with patch.dict(sys.modules, {tts.__name__: tts}):
            with self.assertRaisesRegex(RuntimeError, "OpenRouter audio error: declined"):
                self.node.generate_response(**self.args)
        tts.generate_speech.assert_called_once()

    def test_audio_mode_invalid_speaker_json_fails_before_request(self):
        tts = self.tts_module()
        with patch.dict(sys.modules, {tts.__name__: tts}):
            with self.assertRaisesRegex(RuntimeError, "speaker turn"):
                self.node.generate_response(**self.args, tts_speakers="{not json")
        tts.generate_speech.assert_not_called()

    def test_audio_mode_rejects_chat_media_and_multi_images(self):
        tts = self.tts_module()
        with patch.dict(sys.modules, {tts.__name__: tts}):
            for kwargs in ({"pdf_data": {"bytes": b"pdf"}}, {"audio_data": {}}, {"image_1": object(), "image_2": object()}):
                with self.subTest(kwargs=list(kwargs)):
                    with self.assertRaisesRegex(RuntimeError, "audio_data|Multiple image"):
                        self.node.generate_response(**{**self.args, **kwargs})
        tts.generate_speech.assert_not_called()
        self.module.requests.post.assert_not_called()

    def test_audio_mode_passes_reference_audio_and_image(self):
        tts = self.tts_module()
        import torch as real_torch
        reference = {"waveform": real_torch.zeros((1, 1, 2)), "sample_rate": 16000}
        image = real_torch.zeros((1, 8, 8, 3))
        with patch.dict(sys.modules, {tts.__name__: tts}), patch.object(
                self.module.OpenRouterNode, "image_to_base64", return_value="AAAA"):
            self.node.generate_response(**self.args, tts_reference_audio=reference, tts_reference_image=image)
        kwargs = tts.generate_speech.call_args.kwargs
        self.assertEqual(kwargs["reference_audio"], [reference])
        self.assertEqual(kwargs["reference_image"], "data:image/png;base64,AAAA")

    def test_audio_cache_includes_tts_settings_and_reference_fingerprint(self):
        audio = types.ModuleType(f"{self.package}.openrouter_audio")
        audio.audio_fingerprint = Mock(return_value="clip-fingerprint")
        base = dict(self.args, tts_voice="george", tts_format="mp3", tts_speed=1.5,
                    tts_reference_audio={"bytes": b"clip"})
        with patch.dict(sys.modules, {audio.__name__: audio}):
            first = self.node.IS_CHANGED(**base)
            same = self.node.IS_CHANGED(**base)
            changed = self.node.IS_CHANGED(**{**base, "tts_voice": "sarah"})
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)
        self.assertNotIn("test-key", repr(first))

    def test_audio_mode_returns_none_audio_on_null_response(self):
        # Defensive: a None waveform from the TTS module must raise, never
        # reach PreviewAudio as a successful cached None.
        tts = self.tts_module(result=(None, "gen-1"))
        with patch.dict(sys.modules, {tts.__name__: tts}):
            with self.assertRaisesRegex(RuntimeError, "no usable audio"):
                self.node.generate_response(**self.args)


if __name__ == "__main__":
    unittest.main()
