# OpenRouter Load Audio File

Loads an audio file from ComfyUI's `input` folder as raw, unmodified bytes (no decode/re-encode roundtrip), for the OpenRouter node's **audio_data** input.

The built-in **Load Audio** node decodes the file to a waveform, which the OpenRouter node then has to re-encode (as WAV, or MP3 for long clips) before upload - inflating size and, if re-encoded as MP3, adding a second lossy compression pass. This node instead sends the original file exactly as-is: smaller, faster, and lossless.

Recommended for long clips such as full songs.

## Inputs

- **audio**: Audio file to load, picked from ComfyUI's `input` folder. Upload a file or choose one already present.

## Outputs

- **audio_data**: A `{filename, bytes}` dict carrying the file's original, unmodified contents. Connect directly to the OpenRouter node's **audio_data** input.
