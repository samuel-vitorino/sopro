https://github.com/user-attachments/assets/40254391-248f-45ff-b9a4-107d64fbb95f

# Sopro TTS

[![Alt Text](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/samuel-vitorino/sopro)

### 📰 News

**2026.02.04 - SoproTTS v1.5 is out: more stable, faster, and smaller. Trained for just $100, it reaches 250 ms TTFA streaming and 0.05 RTF (~20× realtime) on CPU.**

Sopro (from the Portuguese word for “breath/blow”) is a lightweight English text-to-speech model I trained as a side project. Sopro is composed of dilated convs (à la WaveNet) and lightweight cross-attention layers, instead of the common Transformer architecture. Even though Sopro is not SOTA across most voices and situations, I still think it’s a cool project made with a very low budget (trained on a single L40S GPU), and it can be improved with better data.

Some of the main features are:

- **147M parameters**
- **Streaming**
- **Zero-shot voice cloning**
- **0.05 RTF on CPU** (measured on an M3 base model), meaning it generates 32 seconds of audio in 1.77 seconds
- **3-12 seconds of reference audio** for voice cloning

---

## Instructions

I only pinned the minimum dependency versions so you can install the package without having to create a separate env. However, some versions of Torch work best. For example, on my M3 CPU, `torch==2.10.0` (without `torchvision`) achieves ~600 it/s on the AR generation.

(Optional)

```bash
conda create -n soprotts python=3.10
conda activate soprotts
```

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
soprotts \
  --text "Sopro is a lightweight 169 million parameter text-to-speech model. Some of the main features are streaming, zero-shot voice cloning, and 0.25 real-time factor on the CPU." \
  --ref_audio ref.wav \
  --out out.wav
```

You have the expected `temperature` and `top_p` parameters, alongside:

- `--style_strength` (controls the FiLM strength; increasing it can improve or reduce voice similarity; default `1.2`)

### Python

#### Non-streaming

```python
from sopro import SoproTTS

tts = SoproTTS.from_pretrained("samuel-vitorino/sopro", device="cpu")

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

tts = SoproTTS.from_pretrained("samuel-vitorino/sopro", device="cpu")

chunks = []
for chunk in tts.stream(
    "Hello! This is a streaming Sopro TTS example.",
    ref_audio_path="ref.mp3",
):
    chunks.append(chunk.cpu())

wav = torch.cat(chunks, dim=-1)
tts.save_wav("out_stream.wav", wav)
```

You can also precalculate the reference to reduce TTFA:

```python
import torch
from sopro import SoproTTS

tts = SoproTTS.from_pretrained("samuel-vitorino/sopro", device="cpu")

ref = tts.prepare_reference(ref_audio_path="ref.mp3")

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

## Interactive streaming demo

![Screenshot](https://github.com/user-attachments/assets/a1902bb9-734c-4da8-ad0d-f842fb7da370)

After you install the `sopro` package:

```bash
pip install -r demo/requirements.txt
uvicorn demo.server:app --host 0.0.0.0 --port 8000
```

Or with docker:

```bash
docker build -t sopro-demo .
docker run --rm -p 8000:8000 sopro-demo
```

Navigate to http://localhost:8000 on your browser.

---

## Disclaimers

- Sopro can be inconsistent, so mess around with the parameters until you get a decent sample.
- Voice cloning is **highly dependent** on mic quality, ambient noise, etc. On more OOD voices it might fail to match the voice well.
- Prefer words instead of abbreviations and symbols. For example, `“1 + 2”` → `“1 plus 2”`. That said, Sopro can generally read abbreviations like “CPU”, “TTS”, etc.
- The streaming version is not bit-exact compared to the non-streaming version. For best quality, prioritize the non-streaming version.
- If you use torchaudio to read or write audio, ffmpeg may be required. I recommend just using soundfile.
- I will publish the training code once I have time to organize it.

Currently, generation is limited to **~32 seconds (400 frames)**. You can increase it, but the model generally hallucinates beyond that.

AI was used mainly for creating the web demo, organizing my messy code into this repo, ablations and brainstorming.

I would love to support more languages and continue improving the model. If you like this project, consider buying me a coffee so I can buy more compute: https://buymeacoffee.com/samuelvitorino

---

## Training data

- [Emilia YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- [LibriTTS-R](https://huggingface.co/datasets/mythicinfinity/libritts_r)
- [Mozilla Common Voice 22](https://datacollective.mozillafoundation.org/)
- [MLS](https://huggingface.co/datasets/parler-tts/mls_eng_10k)

---

## Acknowledgements

- [Mimi Codec (Kyutai)](https://huggingface.co/kyutai/mimi)
- [WaveNet](https://arxiv.org/abs/1609.03499)
- [Attentive Stats Pooling](https://arxiv.org/abs/1803.10963)
- [AudioLM](https://arxiv.org/pdf/2209.03143)
- [CSM](https://github.com/SesameAILabs/csm)
- [Llama 3.2 (Tokenizer)](https://huggingface.co/meta-llama/Llama-3.2-1B)
