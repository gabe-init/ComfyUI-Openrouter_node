"""No-spend regression tests for the TTS request contract and audio decoding."""

import base64
import io
import struct
import unittest
import wave

import torch
from unittest.mock import Mock, patch

import numpy as np
import requests

import openrouter_catalog as catalog
import openrouter_tts as tts


class FakeResponse:
    def __init__(self, status=200, content=b"", headers=None, json_data=None):
        self.status_code = status
        self.content = content
        self.headers = headers or {}
        self._json = json_data

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json


def wav_bytes(samples=(0, 100, -100), rate=24000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


def metadata(voices=("george", "sarah")):
    return {"id": "vendor/voice", "architecture": {"output_modalities": ["speech"]},
            "supported_voices": list(voices)}


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.require = patch.object(tts, "_require_metadata",
                                    return_value={"id": "vendor/voice", "supported_voices": None}).start()
        self.addCleanup(patch.stopall)

    def payload(self, **kwargs):
        kwargs.setdefault("model", "vendor/voice")
        kwargs.setdefault("prompt", "Hello there")
        return tts.build_payload(**kwargs)

    def test_minimal_request_sends_model_and_input_only(self):
        self.assertEqual(self.payload(), {"model": "vendor/voice", "input": "Hello there"})
        self.require.assert_called_once_with("vendor/voice")

    def test_voice_format_instructions_and_speed(self):
        self.assertEqual(
            self.payload(voice="george", response_format="mp3", instructions="warm", speed=1.5),
            {"model": "vendor/voice", "input": "Hello there", "voice": "george",
             "response_format": "mp3", "instructions": "warm", "speed": 1.5},
        )

    def test_unsupported_voice_fails_before_payment(self):
        self.require.return_value = metadata()
        with self.assertRaisesRegex(ValueError, "does not support voice 'nobody'"):
            self.payload(voice="nobody")
        with self.assertRaises(ValueError):
            tts.generate_speech("key", "vendor/voice", "hi", voice="nobody")

    def test_auto_voice_with_known_model_but_no_default_fails_locally(self):
        self.require.return_value = metadata()
        with self.assertRaisesRegex(ValueError, "requires an explicit voice.*george, sarah"):
            self.payload(voice="auto")

    def test_auto_omits_voice_and_format(self):
        # Models without a voice list (or @presets) may omit the voice.
        self.assertNotIn("voice", self.payload(voice="auto"))
        self.assertNotIn("response_format", self.payload(response_format="auto"))

    def test_bad_format_and_speed_rejected_before_discovery(self):
        for kwargs in ({"response_format": "ogg"}, {"speed": 9}, {"speed": "fast"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.payload(**kwargs)

    def test_preset_model_skips_discovery_and_sends_unverified_voice(self):
        payload = tts.build_payload("@preset/my-tts", "Hi", voice="alloy")
        self.assertEqual(payload, {"model": "@preset/my-tts", "input": "Hi", "voice": "alloy"})
        self.require.assert_not_called()

    def test_empty_list_metadata_sends_voice_unverified(self):
        self.require.return_value = {"id": "vendor/voice", "supported_voices": None}
        self.assertEqual(self.payload(voice="custom-voice")["voice"], "custom-voice")

    def test_multi_speaker_turns_replace_prompt_and_validate_voices(self):
        payload = self.payload(voice="george", speaker_turns=[
            {"text": "Hi Jane", "voice": "sarah", "instructions": "tired"},
            {"text": "Hey"},
        ])
        self.assertEqual(payload["input"], [
            {"text": "Hi Jane", "voice": "sarah", "instructions": "tired"},
            {"text": "Hey", "voice": "george"},
        ])
        self.assertNotIn("voice", payload, "top-level voice must not override per-turn voices")

    def test_turns_without_own_voice_inherit_top_level(self):
        payload = self.payload(voice="george", speaker_turns=[{"text": "Hey"}])
        self.assertEqual(payload["input"], [{"text": "Hey", "voice": "george"}])

    def test_invalid_turns_rejected_before_payment(self):
        self.require.reset_mock()
        self.require.return_value = metadata()
        for turns in ([""], [{"voice": "sarah"}], [{"text": "ok", "voice": "bogus"}], "not-a-list",
                      [{"text": 1}], [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"},
                                      {"text": "e"}, {"text": "f"}, {"text": "g"}, {"text": "h"},
                                      {"text": "i"}, {"text": "j"}, {"text": "k"}]):
            with self.subTest(turns=turns):
                with self.assertRaises(ValueError):
                    self.payload(speaker_turns=turns)
        self.require.assert_called()

    def test_reference_audio_becomes_input_audio_part(self):
        clip = wav_bytes()
        payload = self.payload(reference_audio=[{"filename": "ref.wav", "bytes": clip}])
        references = payload["input_references"]
        self.assertEqual(references[0]["type"], "input_audio")
        self.assertTrue(references[0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
        base64.b64decode(references[0]["input_audio"]["data"].split(",", 1)[1], validate=True)

    def test_reference_mp3_file_bytes_pass_through_unmodified(self):
        # The whole point of OpenRouter Load Audio File: original compressed
        # bytes (e.g. a full MP3 song) reach OpenRouter byte-identical - no
        # decode/re-encode roundtrip, no 15 MB WAV growth.
        mp3 = b"ID3" + b"\x00" * 7 + b"\x00" * 20 + b"\xff\xfb\x90\x64" + b"payload"
        payload = self.payload(reference_audio=[{"filename": "song.mp3", "bytes": mp3}])
        header, b64 = payload["input_references"][0]["input_audio"]["data"].split(";base64,", 1)
        self.assertEqual(header, "data:audio/mp3")
        self.assertEqual(base64.b64decode(b64, validate=True), mp3)

    def test_reference_native_oversized_clip_compresses_to_mp3(self):
        # Native waveforms follow the same rule as chat: a WAV above ~15 MB is
        # MP3-compressed instead of risking OpenRouter's request size limit.
        samples = 15 * 1024 * 1024 // 4 + 1000  # stereo PCM16 = 4 bytes/sample
        clip = {"waveform": torch.zeros((1, 2, samples)), "sample_rate": 44100}
        payload = self.payload(reference_audio=[clip])
        header = payload["input_references"][0]["input_audio"]["data"].split(";base64,", 1)[0]
        self.assertEqual(header, "data:audio/mp3")

    def test_reference_text_follows_single_clip(self):
        clip = {"filename": "ref.wav", "bytes": wav_bytes()}
        payload = self.payload(reference_audio=[clip], reference_text="This is the transcript.")
        self.assertEqual(payload["input_references"], [
            {"type": "input_audio", "input_audio": {"data": payload["input_references"][0]["input_audio"]["data"]}},
            {"type": "text", "text": "This is the transcript."},
        ])

    def test_reference_text_with_multiple_clips_rejected(self):
        clip = {"filename": "ref.wav", "bytes": wav_bytes()}
        with self.assertRaisesRegex(ValueError, "transcript"):
            self.payload(reference_audio=[clip, clip], reference_text="transcript")

    def test_reference_text_without_clip_rejected(self):
        with self.assertRaisesRegex(ValueError, "transcript"):
            self.payload(reference_text="transcript")

    def test_reference_limits_and_mixing(self):
        clip = {"filename": "ref.wav", "bytes": wav_bytes()}
        with self.assertRaisesRegex(ValueError, "at most 3"):
            self.payload(reference_audio=[clip] * 4)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.payload(reference_audio=[clip], reference_image="https://example.com/a.png")

    def test_reference_text_is_sent_next_to_the_clip(self):
        # The transcript lives next to the clip in OpenRouter's schema.
        clip = {"filename": "ref.wav", "bytes": wav_bytes()}
        payload = self.payload(reference_audio=[clip])
        self.assertEqual(len(payload["input_references"]), 1)

    def test_prompt_limits(self):
        with self.assertRaisesRegex(ValueError, "3000"):
            self.payload(prompt="x" * 3001)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.require = patch.object(tts, "_require_metadata",
                                    return_value={"id": "vendor/voice", "supported_voices": None}).start()
        self.http = Mock()
        self.addCleanup(patch.stopall)

    def generate(self, response=None, **kwargs):
        if response is not None:
            self.http.post.return_value = response
        args = dict(api_key="secret-key", model="vendor/voice", prompt="hi", _http=self.http)
        args.update(kwargs)
        return tts.generate_speech(**args)

    def test_happy_path_posts_raw_bytes_and_returns_generation_id(self):
        pcm = struct.pack("<3h", 0, 16384, -16384)
        audio, generation_id = self.generate(
            FakeResponse(content=pcm, headers={"Content-Type": "audio/pcm", "X-Generation-Id": "gen-1"}))
        self.assertEqual(generation_id, "gen-1")
        self.assertEqual(audio["sample_rate"], tts.DEFAULT_SAMPLE_RATE)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 1, 3))
        self.assertAlmostEqual(audio["waveform"][0, 0, 1].item(), 0.5, places=5)
        call = self.http.post.call_args
        self.assertEqual(call.args[0], catalog.API_BASE + "/audio/speech")
        self.assertEqual(call.kwargs["json"]["input"], "hi")
        self.assertFalse(call.kwargs["allow_redirects"])

    def test_wav_response_decodes_to_native_audio(self):
        data = wav_bytes((0, 222, -222), rate=44100)
        audio, _ = self.generate(FakeResponse(content=data, headers={"Content-Type": "audio/wav"}))
        self.assertEqual(audio["sample_rate"], 44100)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 1, 3))
        self.assertAlmostEqual(audio["waveform"][0, 0, 1].item(), 222 / 32768, places=5)

    def test_error_response_never_leaks_key_and_does_not_retry(self):
        self.http.post.return_value = FakeResponse(status=402, json_data={"error": {"message": "insufficient credits secret-key"}})
        with self.assertRaisesRegex(RuntimeError, "402") as caught:
            self.generate()
        self.assertNotIn("secret-key", str(caught.exception))
        self.http.post.assert_called_once()

    def test_transport_failure_reports_no_retry(self):
        self.http.post.side_effect = requests.Timeout("secret-key boom")
        with self.assertRaisesRegex(RuntimeError, "no automatic retry") as caught:
            self.generate()
        self.assertNotIn("secret-key", str(caught.exception))

    def test_json_error_document_instead_of_audio_fails_clearly(self):
        response = FakeResponse(content=b'{"error": {}}', headers={"Content-Type": "application/json"})
        with self.assertRaisesRegex(ValueError, "error document"):
            self.generate(response)

    def test_empty_audio_fails(self):
        with self.assertRaisesRegex(ValueError, "empty audio"):
            self.generate(FakeResponse(content=b"", headers={"Content-Type": "audio/mpeg"}))


class DecodeTests(unittest.TestCase):
    def test_odd_pcm_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            tts._decode_audio(b"\x01\x02\x03", "audio/pcm")

    def test_container_without_audio_stream_rejected(self):
        with self.assertRaises(ValueError):
            tts._decode_audio(b"not audio at all", "audio/mpeg")


class CatalogTestsForSpeech(unittest.TestCase):
    def test_speech_catalog_reads_output_modalities_filter(self):
        with catalog._condition:
            catalog._records = {kind: [] for kind in catalog.CATALOG_URLS}
            catalog._success_at = {kind: None for kind in catalog.CATALOG_URLS}
            catalog._attempt_at = {kind: None for kind in catalog.CATALOG_URLS}
            catalog._errors = {}
        with catalog._endpoint_condition:
            catalog._endpoints.clear()
        with patch.object(catalog, "_read_json", return_value={"data": [metadata()]}) as network:
            catalog.refresh_catalog()
            speech_calls = [call for call in network.call_args_list
                            if call.args[0] == catalog.API_BASE + "/models?output_modalities=speech"]
            self.assertEqual(len(speech_calls), 1)
            self.assertEqual(catalog.require_model("speech", "vendor/voice"), metadata())
        with self.assertRaisesRegex(ValueError, "Catalog kind"):
            catalog.model_ids("nope")


if __name__ == "__main__":
    unittest.main()
