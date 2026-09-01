# Sopro web demo

Shared frontend for the Python and ONNX demos. The Python package serves it with
`soprotts serve`. The standalone server below connects it to the browser ONNX
runtime, selecting WebGPU on supported desktops and WASM/SIMD elsewhere.

Run:

```bash
python serve.py
```

Test a local export with the local runtime source:

```bash
python serve.py --model-dir /path/to/web-model
```

For testing from another device on your network, serve over HTTPS:

```bash
python serve.py \
  --host 0.0.0.0 \
  --cert /path/to/certificate.crt \
  --key /path/to/private.key
```

Open the printed URL, choose or record a short voice reference, enter text, and
press Speak. Language detection is automatic unless a language is selected in
Voice controls. The package and model are loaded from their published defaults;
the server supplies the isolation headers required by the demo.
