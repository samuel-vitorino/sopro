<p align="center">
  <img src="assets/banner.svg" alt="Sopro V2" width="100%">
</p>

# Sopro TTS

[![Blog](https://img.shields.io/badge/Blog-Sopro%20V2-11110f?logo=rss&logoColor=white)](https://research.haloneuro.ai/posts/sopro-v2)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/samuel-vitorino/sopro-v2-turbo)
[![ONNX Demo](https://img.shields.io/badge/Demo-In--browser-005ced?logo=onnx&logoColor=white)](https://samuel-vitorino.github.io/sopro/)

> **Compatibility:** Current model artifacts require `sopro>=2.2.0` or `@soprotts/onnx-web>=0.3.0`; upgrade with `pip install -U sopro` or `npm install @soprotts/onnx-web@latest`.

Sopro (from the Portuguese word for "breath/blow") is a lightweight voice-cloning text-to-speech model family. This repo ships **sopro-v2-turbo**, a 120M-parameter open model that streams, runs comfortably on a laptop CPU or in the browser, and reaches SOTA-level intelligibility against much larger systems. The full story, evaluations, and audio samples are in the [blog post](https://research.haloneuro.ai/posts/sopro-v2).

Main features:

- **120M parameters**
- **English, European Portuguese, French, and German**
- **Streaming** with ~300 ms time-to-first-audio on a laptop CPU
- **Zero-shot voice cloning** from 5-20 seconds of reference audio
- **0.24 RTF offline / 0.21 RTF streaming** on an M3 CPU, 0.07 RTF on an H100
- **Runs in the browser** via an ONNX runtime

---

## Local demo

Run the model and open the demo with one command:

```bash
uvx --from sopro soprotts serve
```

If Sopro is already installed:

```bash
soprotts serve
```

Then navigate to http://localhost:7860. The model downloads on first use and stays in the Hugging Face cache. Sopro selects CUDA or CPU automatically and defaults to CPU on macOS (pass `--device mps` explicitly to use MPS). Use `soprotts serve --help` for model, device, port, and CPU int8 options.

## Browser demo

There is also a fully in-browser demo with no server involved: https://samuel-vitorino.github.io/sopro/. On mobile the model is quantized, so results can be slightly below the local demo, and devices with low memory may crash. The dependency-isolated ONNX runtime and exporter are documented in [`web/README.md`](web/README.md). Both demos use the frontend in [`demos/web`](demos/web).

---

## Installation

### From PyPI

```bash
pip install -U sopro
```

### From the repo

```bash
git clone https://github.com/samuel-vitorino/sopro
cd sopro
pip install -e .
```

---

## Examples

### CLI

```bash
soprotts "Sopro is a lightweight 120 million parameter text-to-speech model that streams and runs on device." --ref ref.wav --out out.wav
```

Add `--stream` for the streaming path. You have the expected `--temperature`, `--top-p`, and `--top-k` parameters, alongside:

- `--lang` (`en`, `pt`, `fr`, `de`; optional, helps pronunciation on ambiguous text)
- `--int8` (int8 AR weights on CPU)
- `--steps` (acoustic solver steps; default 2). If you want to trade speed for quality, 8, 16, or even 32 steps can give higher quality speech on more challenging references
- `--max-seconds` (cap per generated segment; long text is split into segments, so total length is unbounded)

### Python

#### Non-streaming

```python
from sopro import SoproTTS

tts = SoproTTS.from_pretrained("samuel-vitorino/sopro-v2-turbo", device="cpu")

wav = tts.synthesize(
    "Hello! This is a non-streaming Sopro TTS example.",
    ref_audio_path="ref.wav",
)

tts.save_wav("out.wav", wav)
```

#### Streaming

```python
import torch
from sopro import SoproTTS

tts = SoproTTS.from_pretrained("samuel-vitorino/sopro-v2-turbo", device="cpu")

chunks = []
for chunk in tts.stream(
    "Hello! This is a streaming Sopro TTS example.",
    ref_audio_path="ref.mp3",
):
    chunks.append(chunk.cpu())

wav = torch.cat(chunks, dim=-1)
tts.save_wav("out_stream.wav", wav)
```

`chunk_frames` defaults to 64 and must be at least 64. Larger values such as 128 or 256 are supported and reduce call frequency at the cost of later audio emission.

You can also precalculate the reference to reduce time-to-first-audio:

```python
import torch
from sopro import SoproTTS

tts = SoproTTS.from_pretrained("samuel-vitorino/sopro-v2-turbo", device="cpu")

ref = tts.prepare_reference(ref_audio_path="ref.mp3", stream=True)

chunks = []
for chunk in tts.stream(
    "Hello! This is a streaming Sopro TTS example.",
    ref=ref,
):
    chunks.append(chunk.cpu())

wav = torch.cat(chunks, dim=-1)
tts.save_wav("out_stream.wav", wav)
```

---

## Disclaimers

- We did not add watermarking: with an open-source inference pipeline it would be trivial to remove, so it would only provide a false sense of safety. Please use the model for good: do not impersonate people.
- The text frontend is deliberately minimal, so some abbreviations, numbers, and symbols may not be pronounced correctly. Prefer words: `1 + 2` should be written `one plus two`. That said, Sopro generally reads common abbreviations like "CPU" or "TTS" fine, and you can put a language-specific normalizer in front of it.
- Mixed-language text is a weak spot: words from one language inside a sentence of another (an English product name in a Portuguese sentence, for example) can be mispronounced.
- The streaming path (chunked attention and the causal vocoder) is not bit-exact with the offline path. For best quality, prefer the offline path.
- We are not planning to release the training code in the near future due to its complexity.

---

## Training data

- [Emilia YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- [LibriTTS-R](https://huggingface.co/datasets/mythicinfinity/libritts_r)
- [FalAR](https://arxiv.org/abs/2605.27062)

---

## Acknowledgements

- [CSM](https://github.com/SesameAILabs/csm)
- [F5-TTS](https://github.com/SWivid/F5-TTS)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Vocos](https://github.com/gemelo-ai/vocos)
- [Whisper](https://github.com/openai/whisper)
