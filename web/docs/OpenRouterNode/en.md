# OpenRouter LLM Node

Interact with OpenRouter's API for chat, image, and video generation, including multimodal inputs (images, PDFs, audio) and model-aware controls.

## Inputs

- **api_key**: Your OpenRouter API key. Leave blank to load it from `openrouter_api_key.json` or the `LLM_KEY` environment variable.
- **system_prompt** / **user_message_box**: Chat prompt. **user_message_input** can override the message box from another node.
- **model**: Fetched live from OpenRouter's chat/image/video catalogs.
- **request_type**: `chat` (default), `image`, or `video` - switches which fields below apply.
- **web_search** / **cheapest** / **fastest**: Append `:online` / `:floor` / `:nitro` routing modifiers (chat).
- **aspect_ratio** / **image_resolution** / **image_quality** / **image_background**: Image generation controls, filtered by the selected model's capabilities.
- **reasoning_effort** / **service_tier**: Chat-only reasoning and routing tier controls.
- **seed**, **temperature**, **request_timeout**: Standard generation controls.
- **pdf_data** / **pdf_engine**: PDF document input and OCR engine (chat).
- **audio_data** / **audio_encoding**: Audio input (native AUDIO or a `{filename, bytes}` dict, e.g. from **OpenRouter Load Audio File**) and its upload encoding (chat).
- **image_1**...**image_10**: Multiple image inputs; additional slots appear as you connect images.
- **chat_mode**: Maintain conversation context across runs.
- **video_mode** / **video_duration** / **video_resolution** / **video_generate_audio** / **video_wait_timeout** / **video_job_id**: Video generation and job-recovery controls.

## Outputs

- **Output**: The text response.
- **image**: Generated/returned image, or an empty tensor.
- **Stats**: Token/cost/model statistics.
- **Credits**: Remaining OpenRouter account balance.
- **video**: Native ComfyUI VIDEO output (requires ComfyUI 0.3.31+ with PyAV).

See the repository README for full details, including Chat Mode session management and video job recovery.
