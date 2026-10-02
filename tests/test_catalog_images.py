"""Discovery and image API regressions; no ComfyUI, GPU, or paid calls required."""

import base64
import threading
import time
import unittest
from unittest.mock import patch

import requests

import openrouter_catalog as catalog
import openrouter_images as images


PNG = b"\x89PNG\r\n\x1a\nimage-test-fixture"


def enum(*values):
    return {"type": "enum", "values": list(values)}


def endpoint(tag="test", **parameters):
    return {"provider_tag": tag, "supported_parameters": parameters}


class CatalogTests(unittest.TestCase):
    def setUp(self):
        with catalog._condition:
            self.assertFalse(catalog._refreshing)
            catalog._records = {kind: [] for kind in catalog.CATALOG_URLS}
            catalog._success_at = {kind: None for kind in catalog.CATALOG_URLS}
            catalog._attempt_at = {kind: None for kind in catalog.CATALOG_URLS}
            catalog._errors = {}
        with catalog._endpoint_condition:
            self.assertFalse(catalog._endpoint_inflight)
            catalog._endpoints.clear()
            catalog._endpoint_attempt_at.clear()
            catalog._endpoint_errors.clear()

    def test_catalogs_keep_distinct_raw_capabilities_and_independent_snapshots(self):
        records = {
            "chat": [{"id": "vendor/shared", "supported_parameters": ["temperature"]}],
            "image": [{"id": "vendor/shared", "supported_parameters": {"quality": enum("high")}}],
            "video": [{"id": "vendor/movie", "pricing_skus": {"video_tokens_4k": {"price": "3"}}}],
        }
        by_url = {catalog.CATALOG_URLS[kind]: {"data": value} for kind, value in records.items()}
        with patch.object(catalog, "_read_json", side_effect=lambda url: by_url[url]) as read:
            first = catalog.refresh_catalog()
            self.assertEqual({kind: first[kind] for kind in records}, records)
            first["image"][0]["supported_parameters"].clear()
            self.assertEqual(catalog.get_model("image", "vendor/shared"), records["image"][0])
            self.assertEqual(catalog.model_ids("chat"), ["vendor/shared"])
            self.assertEqual(read.call_count, 3)
            self.assertIsNotNone(first["updated_at"])

    def test_snapshot_returns_while_one_background_refresh_is_waiting(self):
        entered = threading.Event()
        release = threading.Event()

        def read(url):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test worker did not release")
            return {"data": [{"id": "vendor/model"}]}

        with patch.object(catalog, "_read_json", side_effect=read) as network:
            started = time.monotonic()
            initial = catalog.get_catalog()
            self.assertLess(time.monotonic() - started, 0.5)
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(initial["image"], [])
                for _ in range(10):
                    self.assertEqual(catalog.get_catalog(refresh=True)["video"], [])
                self.assertEqual(network.call_count, 1)
            finally:
                release.set()
                completed = catalog.refresh_catalog()
            self.assertEqual(network.call_count, 3)
            self.assertEqual(completed["image"][0]["id"], "vendor/model")

    def test_success_ttl_failure_backoff_and_stale_retention(self):
        with patch.object(catalog.time, "time", return_value=1000) as clock:
            with patch.object(catalog, "_read_json", return_value={"data": [{"id": "vendor/old"}]}) as network:
                catalog.refresh_catalog()
                clock.return_value = 1899
                catalog.refresh_catalog()
                self.assertEqual(network.call_count, 3)
                clock.return_value = 1900
                network.side_effect = requests.Timeout("secret response or URL")
                failed = catalog.refresh_catalog()
                self.assertEqual(network.call_count, 6)
                self.assertEqual(failed["image"][0]["id"], "vendor/old")
                self.assertIn("Timeout", failed["errors"]["image"])
                self.assertNotIn("secret", str(failed))
                clock.return_value = 1929
                catalog.refresh_catalog(force=True)
                self.assertEqual(network.call_count, 6)
                clock.return_value = 1930
                network.side_effect = None
                network.return_value = {"data": []}
                recovered = catalog.refresh_catalog()
                self.assertEqual(network.call_count, 9)
                self.assertEqual(recovered["errors"], {})
                self.assertEqual(recovered["image"], [])

    def test_bad_catalog_is_reported_and_missing_model_is_actionable(self):
        with patch.object(catalog, "_read_json", return_value={"data": [{"name": "missing id"}]}):
            snapshot = catalog.refresh_catalog()
            self.assertEqual(set(snapshot["errors"]), {"chat", "image", "video"})
            with self.assertRaisesRegex(ValueError, "not in the available catalog.*discovery"):
                catalog.require_model("image", "vendor/model")
        with self.assertRaisesRegex(ValueError, "Catalog kind"):
            catalog.model_ids("unknown")

    def test_video_selector_excludes_other_operations_without_losing_raw_records(self):
        generation = {"id": "vendor/video", "supported_frame_images": ["first_frame"]}
        records = [generation, {"id": "black-forest-labs/flux-video-edit"},
                   {"id": "runway/aleph-2"}, {"id": "heygen/avatar-iv"},
                   {"id": "vendor/new-upscale", "upscale_factor": {"min": 2, "max": 4}}]
        snapshot = {"video": records}
        self.assertEqual(catalog.video_generation_models(snapshot), [generation])
        self.assertEqual(len(snapshot["video"]), 5)
        self.assertFalse(catalog.is_supported_video_model({}))

    def test_endpoint_cache_uses_definitive_records_and_fails_closed_when_expired(self):
        records = [endpoint(quality=enum("high"))]
        with patch.object(catalog.time, "time", return_value=1000) as clock:
            with patch.object(catalog, "_read_json", return_value={"endpoints": records}) as network:
                first = catalog.image_endpoints("vendor/model")
                first[0]["supported_parameters"].clear()
                self.assertEqual(catalog.image_endpoints("vendor/model"), records)
                network.assert_called_once_with(catalog.API_BASE + "/images/models/vendor/model/endpoints", timeout=20)
                clock.return_value = 1900
                network.side_effect = requests.Timeout("private")
                with self.assertRaisesRegex(ValueError, "Cannot verify image settings"):
                    catalog.image_endpoints("vendor/model")
                with self.assertRaises(ValueError):
                    catalog.image_endpoints("vendor/model")
                self.assertEqual(network.call_count, 2)


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.require = patch.object(catalog, "require_model", return_value={"id": "vendor/image"}).start()
        self.endpoints = patch.object(catalog, "image_endpoints", return_value=[endpoint()]).start()
        self.post = patch.object(images.requests, "post").start()
        self.post.return_value.status_code = 200
        self.post.return_value.json.return_value = {
            "data": [{"b64_json": base64.b64encode(PNG).decode("ascii")}],
            "usage": {"cost": 0.025, "output_tokens": 20},
        }
        self.addCleanup(patch.stopall)

    def generate(self, **kwargs):
        return images.generate_images("test-key", "vendor/image", "test prompt", **kwargs)

    def payload(self):
        return self.post.call_args.kwargs["json"]

    def test_default_request_uses_dedicated_image_contract(self):
        result = self.generate()
        self.require.assert_called_once_with("image", "vendor/image")
        self.assertEqual(self.payload(), {"model": "vendor/image", "prompt": "test prompt"})
        self.assertEqual(self.post.call_args.args, (catalog.API_BASE + "/images",))
        self.assertFalse(self.post.call_args.kwargs["allow_redirects"])
        self.assertEqual(result["images"], [PNG])
        self.assertEqual(result["cost"], 0.025)
        self.assertEqual(result["usage"]["output_tokens"], 20)

    def test_explicit_settings_references_and_seed_are_top_level(self):
        self.endpoints.return_value = [endpoint(
            aspect_ratio=enum("16:9"), resolution=enum("2K"), quality=enum("high"),
            background=enum("opaque"), output_format=enum("jpeg", "png"),
            seed={"type": "boolean"}, input_references={"type": "range", "min": 0, "max": 2},
        )]
        self.generate(reference_urls=["https://example.com/reference.png"], aspect_ratio="16:9", resolution="2K", quality="high", background="opaque", seed=42)
        self.assertEqual(self.payload(), {
            "model": "vendor/image", "prompt": "test prompt", "aspect_ratio": "16:9",
            "resolution": "2K", "quality": "high", "background": "opaque", "seed": 42,
            "input_references": [{"type": "image_url", "image_url": {"url": "https://example.com/reference.png"}}],
            "output_format": "png",
        })

    def test_zero_seed_is_sent_only_when_supported(self):
        self.endpoints.return_value = [endpoint(seed={"type": "boolean"})]
        self.generate()
        self.assertEqual(self.payload()["seed"], 0)

    def test_model_union_does_not_override_endpoint_validation(self):
        self.require.return_value["supported_parameters"] = {"resolution": enum("4K")}
        for setting in ({"resolution": "4K"}, {"seed": 12}):
            with self.subTest(setting=setting):
                with self.assertRaisesRegex(ValueError, "No raster image endpoint"):
                    self.generate(**setting)
        self.post.assert_not_called()

    def test_reference_minimum_and_maximum_are_validated_before_payment(self):
        self.endpoints.return_value = [endpoint(input_references={"type": "range", "min": 1, "max": 1})]
        for refs in ([], ["https://example.com/1.png", "https://example.com/2.png"]):
            with self.subTest(count=len(refs)):
                with self.assertRaisesRegex(ValueError, "reference images"):
                    self.generate(reference_urls=refs)
        self.post.assert_not_called()

    def test_routing_is_restricted_to_compatible_provider(self):
        self.endpoints.return_value = [endpoint("small", resolution=enum("1K")), endpoint("large", resolution=enum("1K", "4K"))]
        self.generate(resolution="4K")
        self.assertEqual(self.payload()["provider"], {"only": ["large"]})

    def test_different_provider_formats_are_not_merged_into_invalid_request(self):
        self.endpoints.return_value = [endpoint("jpeg-only", output_format=enum("jpeg")), endpoint("png-only", output_format=enum("png"))]
        self.generate()
        self.assertEqual(self.payload()["provider"], {"only": ["png-only"]})
        self.assertEqual(self.payload()["output_format"], "png")

    def test_shared_provider_uses_format_supported_by_all_its_endpoints(self):
        self.endpoints.return_value = [endpoint("same", output_format=enum("png", "jpeg")), endpoint("same", output_format=enum("jpeg"))]
        self.generate()
        self.assertEqual(self.payload()["output_format"], "jpeg")
        self.assertNotIn("provider", self.payload())

    def test_ambiguous_shared_provider_tag_cannot_bypass_capabilities(self):
        self.endpoints.return_value = [endpoint("same", quality=enum("low")), endpoint("same", quality=enum("high"))]
        with self.assertRaisesRegex(ValueError, "cannot be selected safely"):
            self.generate(quality="high")
        self.post.assert_not_called()

    def test_vector_and_transparent_jpeg_endpoints_are_rejected(self):
        for params, kwargs in (
            ({"output_format": enum("svg")}, {}),
            ({"output_format": enum("jpeg"), "background": enum("transparent")}, {"background": "transparent"}),
        ):
            with self.subTest(parameters=params):
                self.endpoints.return_value = [endpoint(**params)]
                with self.assertRaisesRegex(ValueError, "No raster image endpoint"):
                    self.generate(**kwargs)
        self.post.assert_not_called()

    def test_invalid_references_are_rejected_without_discovery_or_payment(self):
        for value in ("file:///private.png", "https://user:password@example.com/a", "data:image/png;base64,!!!", "https://example.com:invalid/image.png", "https://example.com/a b.png", "relative.png"):
            with self.subTest(reference=value):
                with self.assertRaises(ValueError):
                    self.generate(reference_urls=[value])
        self.require.assert_not_called()
        self.post.assert_not_called()

    def test_transport_failure_never_retries_or_leaks_exception_body(self):
        self.post.side_effect = requests.Timeout("test-key private prompt data")
        with self.assertRaisesRegex(RuntimeError, "no automatic retry") as caught:
            self.generate()
        self.assertNotIn("test-key", str(caught.exception))
        self.post.assert_called_once()

    def test_http_error_never_reads_or_leaks_response_body(self):
        self.post.return_value.status_code = 400
        self.post.return_value.text = "test-key private prompt data"
        with self.assertRaisesRegex(RuntimeError, "HTTP 400") as caught:
            self.generate()
        self.assertNotIn("private", str(caught.exception))
        self.post.return_value.json.assert_not_called()
        self.post.assert_called_once()

    def test_bad_success_payloads_fail_without_retry(self):
        for index, result in enumerate((
            {}, {"data": []}, {"data": [{"url": "https://example.com/image.png"}]},
            {"data": [{"b64_json": "invalid base64"}]},
            {"data": [{"b64_json": base64.b64encode(b"<svg/>").decode("ascii"), "media_type": "image/svg+xml"}]},
        )):
            with self.subTest(case=index):
                self.post.reset_mock()
                self.post.return_value.json.return_value = result
                with self.assertRaises(RuntimeError):
                    self.generate()
                self.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
