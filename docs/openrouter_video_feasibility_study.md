# OpenRouter Video Extension Feasibility Study

Reference date: 2026-04-23

## Objective

Evaluate how feasible it is to extend `ComfyUI-Openrouter_node` so it can:

- expose more OpenRouter models inside ComfyUI
- support OpenRouter video generation models
- provide a clean path for models such as `alibaba/wan-*`, `bytedance/seedance-*`, `kwaivgi/kling-video-o1`, `openai/sora-2-pro`, and `google/veo-3.1`

This study was completed without running broad paid video generation tests. It is based on:

- analysis of the local repository code
- official OpenRouter documentation
- public OpenRouter model catalog endpoints

## Executive Summary

Yes, this is feasible.

However, it is not just a matter of adding more model IDs to the current list. The existing node is built around `POST /api/v1/chat/completions`, while OpenRouter video generation uses a dedicated asynchronous API:

- `POST /api/v1/videos`
- `GET /api/v1/videos/{jobId}`
- `GET /api/v1/videos/{jobId}/content`

Practical conclusion:

- video support needs its own execution path
- the cleanest design is to keep the current node focused on chat / multimodal text-image-PDF use cases
- a separate video node is the right long-term architecture
- part of the work also involves fixing model discovery, because the original node did not expose the full OpenRouter catalog

## Repository Assessment

### 1. The existing node is a chat node, not a video node

The main execution method in [node.py](K:\Codex_Projects\ComfyUI_OpenRouter\node.py:145) sends requests to:

- `https://openrouter.ai/api/v1/chat/completions`

The response handling only covers:

- text
- generated images returned in `message.images`

The current output signature in [node.py](K:\Codex_Projects\ComfyUI_OpenRouter\node.py:94) is:

- `("STRING", "IMAGE", "STRING", "STRING")`

So the original node had no:

- `VIDEO` output type
- async polling
- final video download flow

### 2. Model loading was incomplete

The original model loading path only queried:

- `https://openrouter.ai/api/v1/models`

without `output_modalities=all`.

According to OpenRouter documentation, `/api/v1/models` defaults to text-output models. Image-only, video, embeddings, rerank, and speech models are not all returned unless the request is broadened.

That explains why some OpenRouter image/video models were missing from the original node list.

### 3. Some UI controls were present but not actually wired

The repository already exposed UI controls such as:

- `aspect_ratio`
- `image_resolution`
- `seed`

But in the original main node implementation, only a subset of parameters were actually sent in the payload. This mattered for image generation, because OpenRouter expects image generation requests to use `modalities` and, for some models, `image_config`.

## What OpenRouter Supports

### 1. Chat multimodal and video generation are different products

OpenRouter currently exposes at least two relevant families:

1. `chat/completions`
   for text, image, audio, PDF, and multimodal requests
2. `videos`
   for actual video generation

This distinction matters because video generation does not happen through standard chat completions.

### 2. Image generation requires explicit `modalities`

The official OpenRouter image generation documentation states that image-capable models must be called with a `modalities` parameter:

- image + text models: `["image", "text"]`
- image-only models like Flux: `["image"]`

This is a key reason why image-only models such as Flux could be present in the OpenRouter catalog but still behave incorrectly unless the request payload is adapted.

### 3. Public video catalog metadata is rich enough for dynamic UI

The public video model catalog exposes useful metadata per model, including:

- supported resolutions
- supported aspect ratios
- supported durations
- frame-image support
- provider passthrough parameters
- pricing metadata

This makes model-aware UI practical.

## Feasibility Verdict

Feasible, with high confidence, if the implementation is split cleanly.

### Straightforward parts

- fixing model discovery through `output_modalities=all`
- discovering image/video-capable models from the public catalog
- adding an image-generation filter in the main node
- adding a dedicated video node
- mapping ComfyUI images to `frame_images` / `input_references`
- implementing async polling + download for videos

### More sensitive parts

- returning a real `VIDEO` object compatible with ComfyUI expectations
- handling provider-specific quirks during final video content download
- avoiding accidental paid executions
- exposing advanced options without cluttering the UI

## Recommended Architecture

### Recommended option: separate responsibilities

Recommended structure:

1. Keep the existing main node for chat / multimodal text-image-PDF use cases
2. Add a dedicated `OpenRouter Video` node
3. Share a catalog helper across both nodes

Why this is preferable:

- different endpoint
- different lifecycle
- different parameters
- different output type
- very different cost profile

## Implemented Direction

The repository work completed so far follows that recommendation:

- the main node now loads models from the broader OpenRouter catalog
- a dedicated `OpenRouter Video` node was added
- a shared `openrouter_catalog.py` helper was introduced
- video widget options are now driven by public model capabilities
- the minimum valid duration is auto-selected when changing video models
- local cost estimation was added from public `pricing_skus`
- final video content download was hardened after a real-world `401` issue during content retrieval

## Main Node Image Generation Improvements

For the main node specifically, the correct direction is:

- allow filtering the model list to image generation models only
- ensure image-capable models send the correct `modalities`
- send `image_config.aspect_ratio` and `image_config.image_size` when relevant

This improves support for:

- Gemini image models
- Flux models
- other image-only or image-generating models exposed by OpenRouter

## Important Limitation

A single universal “transparent background” switch should not be added blindly unless OpenRouter or the provider metadata exposes that capability in a stable and model-specific way. The public catalog is good, but not yet consistent enough for a global assumption there.

## Recommendation

The overall direction remains sound:

1. keep the main node strong for chat + multimodal use cases
2. add targeted image-generation improvements to the main node
3. keep video in a dedicated node
4. treat paid live tests deliberately and conservatively

## Sources

- [OpenRouter Models Guide](https://openrouter.ai/docs/guides/overview/models)
- [OpenRouter Image Generation Guide](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)
- [OpenRouter Chat Completions API](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
- [OpenRouter Video Generation Guide](https://openrouter.ai/docs/guides/overview/multimodal/video-generation)
- [OpenRouter Video Inputs Guide](https://openrouter.ai/docs/guides/overview/multimodal/videos)

## Safety Note

An OpenRouter API key was provided during development discussions, but broad paid testing was intentionally avoided. Live video generation costs can accumulate quickly, so explicit human approval remains the right default for further testing.
