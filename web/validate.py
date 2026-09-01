#!/usr/bin/env python3
"""Check the portable WASM graphs against fp32 shipping-model oracles."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from export import _export_fp32_graphs


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _profile_graph(root: Path, profile_name: str, item: dict) -> Path:
    return (root / profile_name / item.get("url", item["file"])).resolve()


def _validate_manifest_dtypes(root: Path, profiles: dict) -> list[str]:
    failures = []
    expected_types = {"float16": onnx.TensorProto.FLOAT16, "float32": onnx.TensorProto.FLOAT}
    for profile_name, profile in profiles.items():
        for graph_name, item in profile["graphs"].items():
            expected = expected_types.get(item.get("dtype", profile["dtype"]))
            if expected is None:
                continue
            graph = onnx.load_model(_profile_graph(root, profile_name, item), load_external_data=False).graph
            actual = {value.type.tensor_type.elem_type for value in graph.input if value.type.tensor_type.elem_type in expected_types.values()}
            if actual and expected not in actual:
                names = ", ".join(sorted(onnx.TensorProto.DataType.Name(value) for value in actual))
                failures.append(f"{profile_name}/{graph_name} manifest dtype {item.get('dtype', profile['dtype'])} does not match {names}")
    return failures


def _rotary(length: int, dim: int, offset: int = 0):
    positions = np.arange(offset, offset + length, dtype=np.float32)[:, None]
    frequencies = 1.0 / (10000.0 ** ((np.arange(dim) % (dim // 2)) * 2.0 / dim))
    angles = positions * frequencies[None]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _feeds(name: str, cfg, rng: np.random.Generator):
    model = cfg.model
    if name == "reference":
        left = np.arange(20, dtype=np.int64).clip(max=49)
        return {
            "semantic_mel": rng.normal(size=(1, 80, 103)).astype(np.float32),
            "interp_left": left,
            "interp_right": (left + 1).clip(max=50),
            "interp_weight": rng.random(20, dtype=np.float32),
            "speaker_mel": rng.normal(size=(1, 80, 101)).astype(np.float32),
        }
    if name == "semantic_prefill":
        text, style, prompt = 12, 20, 16
        length = int(model.style_prefix_tokens) + text + prompt + 1
        cos, sin = _rotary(length, int(model.ar_model_dim) // int(model.ar_heads))
        bias = np.zeros((1, 1, length, length), np.float32)
        bias[0, 0][np.triu_indices(length, 1)] = -1.0e4
        return {
            "text_ids": rng.integers(0, int(model.text_vocab_size), (1, text), dtype=np.int64),
            "style_tokens": rng.integers(0, int(model.semantic_vocab_size), (1, style), dtype=np.int64),
            "prompt_tokens": rng.integers(0, int(model.semantic_vocab_size), (1, prompt), dtype=np.int64),
            "cos": cos, "sin": sin, "causal_bias": bias,
        }
    if name == "semantic_step":
        past = 37
        cos, sin = _rotary(1, int(model.ar_model_dim) // int(model.ar_heads), past)
        shape = (int(model.ar_blocks), 1, int(model.ar_kv_heads), past, int(model.ar_model_dim) // int(model.ar_heads))
        return {
            "token": rng.integers(0, int(model.semantic_vocab_size), (1, 1), dtype=np.int64),
            "past_k": rng.normal(size=shape).astype(np.float32), "past_v": rng.normal(size=shape).astype(np.float32),
            "cos": cos, "sin": sin,
        }
    if name == "semantic_prefix":
        return {
            "text_ids": rng.integers(0, int(model.text_vocab_size), (1, 12), dtype=np.int64),
            "style_tokens": rng.integers(0, int(model.semantic_vocab_size), (1, 20), dtype=np.int64),
            "prompt_tokens": rng.integers(0, int(model.semantic_vocab_size), (1, 16), dtype=np.int64),
        }
    if name == "semantic_core":
        past, hidden = 37, 1
        head_dim = int(model.ar_model_dim) // int(model.ar_heads)
        cos, sin = _rotary(hidden + 1, head_dim, past)
        shape = (int(model.ar_blocks), 1, int(model.ar_kv_heads), past, head_dim)
        return {
            "hidden": rng.normal(size=(1, hidden, int(model.ar_model_dim))).astype(np.float32),
            "token": rng.integers(0, int(model.semantic_vocab_size), (1, 1), dtype=np.int64),
            "past_k": rng.normal(size=shape).astype(np.float32), "past_v": rng.normal(size=shape).astype(np.float32),
            "cos": cos, "sin": sin, "attention_bias": np.zeros((1, 1, hidden + 1, past + hidden + 1), np.float32),
        }
    if name == "acoustic_condition":
        tokens, frames = 36, 144
        return {
            "semantic_tokens": rng.integers(0, int(model.semantic_vocab_size), (1, tokens), dtype=np.int64),
            "frame_to_token": np.floor(np.arange(frames) * tokens / frames).astype(np.int64),
        }
    if name == "acoustic_offline_2":
        frames = 128
        cos, sin = _rotary(frames, int(model.acoustic_dit_dim_head))
        mask = np.zeros((1, 1, frames), np.float32); mask[:, :, :64] = 1
        return {
            "x_init": rng.normal(size=(1, 100, frames)).astype(np.float32), "mu": rng.normal(size=(1, 100, frames)).astype(np.float32),
            "cond_vec": rng.normal(size=(1, int(model.cond_hidden_dim))).astype(np.float32), "cond_mel": rng.normal(size=(1, 100, frames)).astype(np.float32),
            "cond_mask": mask, "cos": cos, "sin": sin,
        }
    if name == "acoustic_stream_prefill_2":
        frames, chunk = 320, 64
        cos, sin = _rotary(frames, int(model.acoustic_dit_dim_head))
        mask = np.zeros((1, 1, frames), np.float32); mask[:, :, :256] = 1
        feeds = {
            "x_init": rng.normal(size=(1, 100, frames)).astype(np.float32), "mu": rng.normal(size=(1, 100, frames)).astype(np.float32),
            "cond_vec": rng.normal(size=(1, int(model.cond_hidden_dim))).astype(np.float32), "cond_mel": rng.normal(size=(1, 100, frames)).astype(np.float32),
            "cond_mask": mask, "cos": cos, "sin": sin,
        }
        positions = np.arange(frames); ends = np.minimum(frames, (positions // chunk + 1) * chunk)
        feeds["chunk_bias"] = np.where(positions[None] < ends[:, None], 0.0, -1.0e4)[None, None].astype(np.float32)
        return feeds
    if name == "acoustic_stream_step_2":
        new, context, past = 64, 60, 320
        cos, sin = _rotary(new, int(model.acoustic_dit_dim_head), past)
        return {
            "x_init": rng.normal(size=(1, 100, new)).astype(np.float32), "mu_window": rng.normal(size=(1, 100, context + new)).astype(np.float32),
            "cond_vec": rng.normal(size=(1, int(model.cond_hidden_dim))).astype(np.float32), "cond_mel_window": rng.normal(size=(1, 100, context + new)).astype(np.float32),
            "cond_mask_window": rng.integers(0, 2, (1, 1, context + new)).astype(np.float32), "cos": cos, "sin": sin,
            "x_context": rng.normal(size=(2, 1, 100, context)).astype(np.float32),
            "past_k": rng.normal(size=(2, int(model.acoustic_dit_depth), 1, int(model.acoustic_dit_heads), past, int(model.acoustic_dit_dim_head))).astype(np.float16),
            "past_v": rng.normal(size=(2, int(model.acoustic_dit_depth), 1, int(model.acoustic_dit_heads), past, int(model.acoustic_dit_dim_head))).astype(np.float16),
        }
    if name == "vocoder_stream_start":
        return {"mel": rng.normal(size=(1, 100, 64)).astype(np.float32)}
    raise KeyError(name)


def _metrics(expected: np.ndarray, actual: np.ndarray):
    expected, actual = expected.astype(np.float64), actual.astype(np.float64)
    error = np.abs(expected - actual)
    norm = np.linalg.norm(expected.ravel()) * np.linalg.norm(actual.ravel())
    return {"mae": float(error.mean()), "max": float(error.max(initial=0)), "cosine": float(np.dot(expected.ravel(), actual.ravel()) / norm) if norm else 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.web / "manifest.json").read_text())
    profile = manifest["profiles"]["wasm-uint8"]
    webgpu = manifest["profiles"]["webgpu-fp16"]
    rng, report = np.random.default_rng(7), {}
    with tempfile.TemporaryDirectory(prefix="sopro-parity-") as directory:
        oracle = Path(directory)
        cfg, _ = _export_fp32_graphs(args.artifacts.resolve(), oracle, [2])
        parity_feeds = {}
        for name in ("reference", "semantic_prefix", "semantic_core", "acoustic_condition", "acoustic_offline_2", "acoustic_stream_prefill_2", "acoustic_stream_step_2"):
            feeds = _feeds(name, cfg, rng)
            parity_feeds[name] = feeds
            expected = _session(oracle / f"{name}.onnx").run(None, feeds)
            wasm_feeds = {key: value.astype(np.float16) if name.startswith("acoustic_stream_") and key in ("past_k", "past_v") else value for key, value in feeds.items()}
            actual = _session(_profile_graph(args.web, "wasm-uint8", profile["graphs"][name])).run(None, wasm_feeds)
            report[name] = [_metrics(a, b) for a, b in zip(expected, actual)]
            if name == "reference":
                report[name][1]["tokenAgreement"] = float(np.mean(expected[1] == actual[1]))
        feeds = parity_feeds["acoustic_stream_prefill_2"]
        expected = _session(oracle / "acoustic_stream_prefill_2.onnx").run(None, feeds)
        x, contexts, keys, values = feeds["x_init"], [], [], []
        for step in range(manifest["steps"]):
            step_feeds = {"x": x, "x_init": feeds["x_init"], **{key: value for key, value in feeds.items() if key != "x_init"}}
            x, mel, context, key, value = _session(_profile_graph(args.web, "wasm-uint8", profile["graphs"][f"acoustic_stream_prefill_ode_2_{step}"])).run(None, step_feeds)
            contexts.append(context); keys.append(key); values.append(value)
        actual = [mel, np.stack(contexts), np.stack(keys), np.stack(values)]
        report["acoustic_stream_prefill_ode_2"] = [_metrics(a, b) for a, b in zip(expected, actual)]

        feeds = parity_feeds["acoustic_stream_step_2"]
        expected = _session(oracle / "acoustic_stream_step_2.onnx").run(None, feeds)
        x, contexts, keys, values = feeds["x_init"], [], [], []
        past_k, past_v = feeds["past_k"].astype(np.float16), feeds["past_v"].astype(np.float16)
        for step in range(manifest["steps"]):
            step_feeds = {
                "x": x, "x_init": feeds["x_init"], "mu_window": feeds["mu_window"], "cond_vec": feeds["cond_vec"],
                "cond_mel_window": feeds["cond_mel_window"], "cond_mask_window": feeds["cond_mask_window"], "cos": feeds["cos"], "sin": feeds["sin"],
                "x_context": feeds["x_context"][step], "past_k": past_k[step], "past_v": past_v[step],
            }
            x, mel, context, key, value = _session(_profile_graph(args.web, "wasm-uint8", profile["graphs"][f"acoustic_stream_ode_2_{step}"])).run(None, step_feeds)
            contexts.append(context); keys.append(key); values.append(value)
        actual = [mel, np.stack(contexts), np.stack(keys), np.stack(values)]
        report["acoustic_stream_ode_2"] = [_metrics(a, b) for a, b in zip(expected, actual)]
        if "webgpu-fp32" in manifest["profiles"]:
            for name in ("acoustic_condition", "acoustic_offline_2", "acoustic_stream_prefill_2", "acoustic_stream_step_2"):
                feeds = parity_feeds[name]
                expected = _session(oracle / f"{name}.onnx").run(None, feeds)
                actual = _session(_profile_graph(args.web, "webgpu-fp32", manifest["profiles"]["webgpu-fp32"]["graphs"][name])).run(None, feeds)
                report[f"webgpu_fp32_{name}"] = [_metrics(a, b) for a, b in zip(expected, actual)]
        for name in ("reference", "semantic_prefix", "semantic_core"):
            feeds = parity_feeds[name]
            expected = _session(oracle / f"{name}.onnx").run(None, feeds)
            actual = _session(_profile_graph(args.web, "webgpu-fp16", webgpu["graphs"][name])).run(None, feeds)
            key = f"webgpu_mixed_{name}"
            report[key] = [_metrics(a, b) for a, b in zip(expected, actual)]
            if name == "reference":
                report[key][1]["tokenAgreement"] = float(np.mean(expected[1] == actual[1]))
        start_feeds = _feeds("vocoder_stream_start", cfg, rng)
        start = _session(oracle / "vocoder_stream_start.onnx").run(None, start_feeds)
        unified_session = _session(_profile_graph(args.web, "wasm-uint8", profile["graphs"]["vocoder"]))
        actual = unified_session.run(None, {
            "is_start": np.array(True), "is_flush": np.array(False), **start_feeds,
            "embed_state": np.zeros_like(start[1]), "conv_state": np.zeros_like(start[2]),
            "pending": np.zeros_like(start[3]),
        })
        report["vocoder_stream_start"] = [_metrics(a, b) for a, b in zip(start, actual)]
        state = {key: value for key, value in zip(("istft_features", "embed_state", "conv_state", "pending"), start)}
        for name in ("vocoder_stream_step", "vocoder_stream_flush"):
            feeds = {key: state[key] for key in ("embed_state", "conv_state", "pending")}
            if name.endswith("step"): feeds["mel"] = rng.normal(size=(1, 100, 64)).astype(np.float32)
            expected = _session(oracle / f"{name}.onnx").run(None, feeds)
            actual = unified_session.run(None, {
                "is_start": np.array(False), "is_flush": np.array(name.endswith("flush")),
                "mel": feeds.get("mel", np.zeros((1, 100, 0), np.float32)), **feeds,
            })
            if name.endswith("flush"): actual = actual[:1]
            report[name] = [_metrics(a, b) for a, b in zip(expected, actual)]
    failures = _validate_manifest_dtypes(args.web, manifest["profiles"])
    fp32_graphs = {"reference"}
    fp32_graphs.update(name for name in report if name.startswith("webgpu_fp32_"))
    int8_graphs = {"acoustic_condition", "acoustic_offline_2", "acoustic_stream_prefill_2", "acoustic_stream_step_2", "acoustic_stream_prefill_ode_2", "acoustic_stream_ode_2"}
    vocoder_int8 = {"vocoder_stream_start", "vocoder_stream_step", "vocoder_stream_flush"}
    for name in fp32_graphs:
        if any(output["max"] > 1.0e-5 for output in report[name]):
            failures.append(f"{name} is not fp32-equivalent")
    if report["reference"][1]["tokenAgreement"] < 1.0:
        failures.append("reference token output is not fp32-equivalent")
    if any(output["max"] > 1.0e-5 for output in report["webgpu_mixed_reference"]):
        failures.append("WebGPU-route reference graph is not fp32-equivalent")
    if report["webgpu_mixed_reference"][1]["tokenAgreement"] < 1.0:
        failures.append("WebGPU-route reference tokens are not fp32-equivalent")
    for name, outputs in report.items():
        floor = 0.99 if "semantic_" in name else (0.995 if name in int8_graphs else (0.98 if name in vocoder_int8 else 1.0 - 1.0e-7))
        if any(output["cosine"] < floor for output in outputs): failures.append(f"{name} fell below cosine {floor}")
    print(json.dumps({"passed": not failures, "failures": failures, "graphs": report}, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
