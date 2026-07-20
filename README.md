# Imgprompter

A command-line tool built using `charm.land/catwalk` to process images through Vision LLMs (like LLaVA, Qwen-VL, or DeepSeek-VL) via OpenAI-compatible APIs (specifically optimized for `llama.cpp` and `Ollama`).

It generates sidecar `.txt` files for every image processed, making it ideal for bulk image captioning or dataset preparation.

## Features

- **Sidecar Generation**: Automatically saves LLM responses to `filename.txt` corresponding to `filename.jpg`.
- **Reasoning Support**: Full integration with `llama.cpp` reasoning budgets (`off`, `low`, `medium`, `high`, `unlimited`).
- **Thinking Extraction**: Automatically handles models that output reasoning in `<think>` tags, printing thoughts to the terminal (optional) but excluding them from the saved text file.
- **Batch Processing**: Supports file lists and shell globs (e.g., `./images/*.png`).
- **Idempotency**: Optional flag to skip images that already have a corresponding `.txt` file.

## Installation

Ensure you have [Go](https://go.dev/) installed.

```bash
go build
```

## Usage

### Basic Example
Process all JPEGs in a folder using a local Ollama/llama.cpp server:
```bash
./imgprompter -p prompt.txt ./vacation/*.jpg
```

### With Reasoning (Thinking)
Use high-level reasoning for complex images and see the "thought process" in your terminal:
```bash
./imgprompter -p complex-prompt.txt -r high -t ./photos/medical_scan.png
```

### Skip Existing Files
Useful for resuming a large job that was interrupted:
```bash
./imgprompter -s -p caption_prompt.txt "./dataset/**/*"
```

## Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p` | (required) | Path to the text file containing the system prompt or instructions. |
| `-m` | `Qwen-3.7` | The Model ID to request from the server. |
| `-u` | `http://localhost:11434/v1` | Base URL of the OpenAI-compatible API. |
| `-r` | `off` | Reasoning level: `off`, `low` (512 tokens), `medium` (2048), `high` (8192), `unlimited` (-1). |
| `-t` | `false` | If true, prints the model's reasoning/thinking to `stdout`. |
| `-s` | `false` | Skip processing if the `.txt` file already exists. |

## Server Requirements

- **llama.cpp**: Run with `llama-server --jinja` to support template-based reasoning controls.
- **Ollama**: Ensure the model you are using supports Vision (e.g., `llava`, `bakllava`).

## License

MIT
