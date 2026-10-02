import base64
import importlib.util
import io
from pathlib import Path
import struct
import unittest
import wave

import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "openrouter_audio.py"
_SPEC = importlib.util.spec_from_file_location("openrouter_audio_tests", _MODULE_PATH)
audio = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audio)


class AudioTests(unittest.TestCase):
    @staticmethod
    def native(waveform=None, sample_rate=16000):
        return {
            "waveform": waveform if waveform is not None else torch.zeros((1, 1, 16)),
            "sample_rate": sample_rate,
        }

    @staticmethod
    def read_wav(prepared):
        block = prepared["block"]
        assert block["type"] == "input_audio"
        assert block["input_audio"]["format"] == "wav"
        raw = base64.b64decode(block["input_audio"]["data"], validate=True)
        with wave.open(io.BytesIO(raw), "rb") as source:
            return source.getparams(), source.readframes(source.getnframes())

    def test_native_pcm16_roundtrip_preserves_sample_rate_channels_and_amplitude(self):
        waveform = torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]])
        prepared = audio.prepare_audio(self.native(waveform, sample_rate=96000))
        params, pcm = self.read_wav(prepared)
        self.assertEqual((params.nchannels, params.sampwidth, params.framerate, params.nframes), (2, 2, 96000, 3))
        self.assertEqual(struct.unpack("<6h", pcm), (-32768, 32767, 0, 0, 32767, -32768))

    def test_native_mono_and_deterministic_fingerprint(self):
        value = self.native()
        first = audio.prepare_audio(value)
        self.assertEqual(first, audio.prepare_audio(value))
        self.assertEqual(first["fingerprint"], audio.audio_fingerprint(value))
        params, _pcm = self.read_wav(first)
        self.assertEqual((params.nchannels, params.nframes), (1, 16))

    def test_noncontiguous_and_gradient_tensor_is_detached_without_mutation(self):
        waveform = torch.linspace(-1, 1, 32).reshape(1, 2, 16)[:, :, ::2].requires_grad_()
        original = waveform.detach().clone()
        self.assertFalse(waveform.is_contiguous())
        params, _pcm = self.read_wav(audio.prepare_audio(self.native(waveform)))
        self.assertEqual((params.nchannels, params.nframes), (2, 8))
        self.assertTrue(torch.equal(waveform, original))
        self.assertTrue(waveform.requires_grad)

    def test_bfloat16_normalizes_and_hashes_changes(self):
        quiet = self.native(torch.zeros((1, 1, 16), dtype=torch.bfloat16))
        loud = self.native(torch.ones((1, 1, 16), dtype=torch.bfloat16))
        self.assertNotEqual(audio.audio_fingerprint(quiet), audio.audio_fingerprint(loud))
        self.assertEqual(audio.prepare_audio(quiet), audio.prepare_audio(self.native()))

    def test_native_identity_includes_channel_layout_and_sample_rate(self):
        mono = torch.zeros((1, 1, 16))
        mono_key = audio.audio_fingerprint(self.native(mono))
        self.assertNotEqual(mono_key, audio.audio_fingerprint(self.native(mono.reshape(1, 2, 8))))
        self.assertNotEqual(mono_key, audio.audio_fingerprint(self.native(mono, sample_rate=8000)))

    def test_rejects_batches_multichannel_empty_wrong_shape_and_integer_tensor(self):
        for waveform in (
            torch.zeros((2, 1, 16)), torch.zeros((0, 1, 16)),
            torch.zeros((1, 6, 16)), torch.zeros((1, 0, 16)),
            torch.zeros((1, 1, 0)), torch.zeros((1, 16)),
            torch.zeros((1, 1, 16), dtype=torch.int16),
        ):
            with self.subTest(shape=tuple(waveform.shape), dtype=waveform.dtype):
                with self.assertRaises(ValueError):
                    audio.prepare_audio(self.native(waveform))

    def test_rejects_nonfinite_or_unnormalized_samples(self):
        for sample in (float("nan"), float("inf"), -float("inf"), 1.01, -1.01):
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    audio.prepare_audio(self.native(torch.tensor([[[sample]]])))

    def test_rejects_invalid_sample_rates(self):
        for rate in (True, False, 0, -16000, 16000.0, "16000", None, 2**32):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    audio.prepare_audio(self.native(sample_rate=rate))

    def test_raw_file_keeps_original_bytes_and_uses_supported_extension(self):
        raw = base64.b64decode(audio.prepare_audio(self.native())["block"]["input_audio"]["data"])
        value = {"filename": "CLIP.WAV", "bytes": raw}
        prepared = audio.prepare_audio(value)
        self.assertEqual(base64.b64decode(prepared["block"]["input_audio"]["data"]), raw)
        self.assertEqual(prepared["fingerprint"], audio.audio_fingerprint(value))

    def test_explicit_supported_format_without_known_extension(self):
        prepared = audio.prepare_audio({"filename": "clip.bin", "format": " PCM16 ", "bytes": b"\x00\x00"})
        self.assertEqual(prepared["block"]["input_audio"]["format"], "pcm16")
        self.assertEqual(
            prepared,
            audio.prepare_audio({"format": "pcm16", "bytes": b"\x00\x00"}),
        )

    def test_aif_extension_and_header_are_aiff(self):
        prepared = audio.prepare_audio({"filename": "clip.aif", "bytes": b"FORM\x00\x00\x00\x04AIFF"})
        self.assertEqual(prepared["block"]["input_audio"]["format"], "aiff")

    def test_id3_tag_does_not_mislabel_aac_as_mp3(self):
        tagged_aac = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xf1data"
        prepared = audio.prepare_audio({"filename": "clip.aac", "bytes": tagged_aac})
        self.assertEqual(prepared["block"]["input_audio"]["format"], "aac")
        with self.assertRaises(ValueError):
            audio.prepare_audio({"filename": "clip.mp3", "bytes": tagged_aac})

    def test_raw_pcm_samples_are_not_mistaken_for_container_headers(self):
        prepared = audio.prepare_audio({"format": "pcm16", "bytes": b"\xff\xf1"})
        self.assertEqual(prepared["block"]["input_audio"]["format"], "pcm16")

    def test_raw_identity_includes_format_but_not_irrelevant_basename(self):
        first = {"filename": "first.pcm16", "bytes": b"\x00" * 6}
        same = {"filename": "second.pcm16", "bytes": b"\x00" * 6}
        changed = {"filename": "first.pcm24", "bytes": b"\x00" * 6}
        self.assertEqual(audio.audio_fingerprint(first), audio.audio_fingerprint(same))
        self.assertNotEqual(audio.audio_fingerprint(first), audio.audio_fingerprint(changed))

    def test_rejects_unknown_conflicting_or_empty_raw_inputs(self):
        for value in (
            None, "clip.wav", {}, {"filename": "clip.wav", "bytes": b""},
            {"filename": "clip.wav", "bytes": "not bytes"},
            {"filename": "clip.unknown", "bytes": b"data"},
            {"bytes": b"data"}, {"format": "banana", "bytes": b"data"},
            {"format": "", "bytes": b"data"},
            {"filename": "", "format": "wav", "bytes": b"data"},
            {"filename": "clip.mp3", "format": "wav", "bytes": b"data"},
            {"filename": "clip.wav", "bytes": b"fLaCdata"},
            {"filename": "clip.wav", "bytes": b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfbdata"},
            {"format": "pcm16", "bytes": b"\x00"},
            {"format": "pcm24", "bytes": b"\x00\x00"},
            {"waveform": torch.zeros((1, 1, 16))},
            {"sample_rate": 16000},
            dict(self.native(), bytes=b"data"),
        ):
            with self.subTest(keys=list(value) if isinstance(value, dict) else type(value)):
                with self.assertRaises(ValueError):
                    audio.prepare_audio(value)
                with self.assertRaises(ValueError):
                    audio.audio_fingerprint(value)

    def test_common_container_mismatches_are_detected(self):
        for raw in (
            b"RIFF\x00\x00\x00\x00WAVE", b"FORM\x00\x00\x00\x00AIFF",
            b"\x00\x00\x00\x14ftypM4A ", b"OggSdata", b"fLaCdata",
            b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfbdata", b"\xff\xfbdata", b"\xff\xf1data", b"ADIFdata",
        ):
            with self.subTest(header=raw[:4]):
                with self.assertRaises(ValueError):
                    audio.prepare_audio({"format": "mp3" if raw.startswith(b"RIFF") else "wav", "bytes": raw})


if __name__ == "__main__":
    unittest.main()
