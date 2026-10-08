# OpenRouter LLM Node

Interact with OpenRouter's API for chat, image, video, and audio (TTS) generation, including multimodal inputs (images, PDFs, audio) and model-aware controls.

## Inputs

- **api_key**: Your OpenRouter API key. Leave blank to load it from `openrouter_api_key.json` or the `LLM_KEY` environment variable.
- **system_prompt** / **user_message_box**: Chat prompt. **user_message_input** can override the message box from another node. In audio mode this text is synthesized.
- **model**: Fetched live from OpenRouter's chat/image/video/speech catalogs.
- **request_type**: `chat` (default), `image`, `video`, or `audio` - switches which fields below apply. Audio mode shows only the prompt and TTS controls; chat adds conversation/multimodal controls; image and video show their generation controls.
- **web_search** / **cheapest** / **fastest**: Append `:online` / `:floor` / `:nitro` routing modifiers (chat).
- **aspect_ratio** / **image_resolution** / **image_quality** / **image_background**: Image generation controls, filtered by the selected model's capabilities.
- **reasoning_effort** / **service_tier**: Chat-only reasoning and routing tier controls.
- **seed**, **temperature**, **request_timeout**: Standard generation controls.
- **pdf_data** / **pdf_engine**: PDF document input and OCR engine (chat).
- **audio_data** / **audio_encoding**: Audio input (native AUDIO or a `{filename, bytes}` dict, e.g. from **OpenRouter Load Audio File**) and its upload encoding (chat).
- **image_1**...**image_10**: Multiple image inputs; additional slots appear as you connect images.
- **chat_mode**: Maintain conversation context across runs.
- **video_mode** / **video_duration** / **video_resolution** / **video_generate_audio** / **video_wait_timeout** / **video_job_id**: Video generation and job-recovery controls.
- **tts_voice**: Voice identifier for audio (TTS) requests as a dropdown filled from the selected model's `supported_voices` (e.g. `george` for elevenlabs/eleven-v4, `Kore` for Gemini TTS). `auto` omits the parameter and fails locally with the voice list when the provider has no default voice. `@preset/` models cannot be verified and send any typed voice unverified.
- **tts_format**: Output audio encoding for audio (TTS) requests: `auto` (OpenRouter default, currently pcm), `mp3`, or `pcm`.
- **tts_instructions**: Optional delivery instructions (tone, pacing, emotion) for audio (TTS) requests. Supported by OpenAI gpt-4o-mini-tts and Gemini TTS; ignored by others.
- **tts_speed**: Playback speed multiplier (0.25-4.0) for audio (TTS) requests; 1.0 omits the parameter.
- **tts_speakers**: Optional multi-speaker input, one JSON turn per line: `{"text": "Hi!", "voice": "Kore"}`. Currently supported by Gemini TTS models; replaces the plain prompt when nonempty.
- **tts_reference_audio**: One reference clip (up to 3 sent) for stateless voice cloning (audio mode). Requests route only to endpoints supporting cloning; address clips with `@Audio1`...`@Audio3` where supported.
- **tts_reference_image**: One image for voice-design models (audio mode). Cannot be combined with audio references.
- **tts_reference_text**: Optional transcript sent next to a single reference clip (audio mode).

## Outputs

- **Output**: The text response, or a summary line with the TTS model and generation ID in audio mode.
- **image**: Generated/returned image, or an empty tensor.
- **Stats**: Token/cost/model statistics.
- **Credits**: Remaining OpenRouter account balance.
- **video**: Native ComfyUI VIDEO output (requires ComfyUI 0.3.31+ with PyAV).
- **audio**: Native ComfyUI AUDIO output in audio mode - the decoded waveform (mp3/pcm are decoded via PyAV), ready for Save Audio / Preview Audio.

See the repository README for full details, including Chat Mode session management, video job recovery, and the TTS feature matrix (voice cloning, image references, multi-speaker).
