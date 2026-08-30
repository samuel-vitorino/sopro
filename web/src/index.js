import { AssetCache, getCachedReference, putCachedReference } from './cache.js';
import { FINAL_FADE_SECONDS, GATE_HOLD_SECONDS, JOIN_FADE_SECONDS, LEAD_IN_SECONDS, SEGMENT_LEAD_SECONDS, SEGMENT_SKIP_SECONDS, StreamingISTFT, cropAndNormalizeReference, energyFraction, energyOnset, fadeEdges, istft, joinSegments, matchGain, outputGain, referenceFeatures, resample, softLimit, speechOnset, trimLead, trimTrail } from './dsp.js';
import { SentencePieceTokenizer, splitText } from './tokenizer.js';

function joinUrl(base, path) { return `${String(base).replace(/\/$/, '')}/${path}`; }
function versionUrl(url, revision) {
  if (!revision) return url;
  return `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(revision)}`;
}
function fileName(url) { return String(url).split('?', 1)[0].split('/').at(-1); }
function int64(values) { return values instanceof BigInt64Array ? values : BigInt64Array.from(values, (value) => BigInt(value)); }
async function cooperate(parameters, index = 0, interval = 1) {
  if (!parameters.cooperative || (index + 1) % interval !== 0) return;
  await new Promise((resolve) => setTimeout(resolve, 0));
}

function platform() {
  const nav = globalThis.navigator ?? {};
  const mobileWebKit = /iP(?:hone|ad|od)/.test(nav.userAgent ?? '') || (nav.platform === 'MacIntel' && nav.maxTouchPoints > 1);
  return { mobileWebKit, mobile: mobileWebKit || /Android|Mobile/i.test(nav.userAgent ?? '') };
}

function float32ToFloat16(values) {
  const source = values instanceof Float32Array ? values : Float32Array.from(values);
  const bits = new Uint32Array(source.buffer, source.byteOffset, source.length), out = new Uint16Array(source.length);
  for (let i = 0; i < source.length; i++) {
    const x = bits[i], sign = (x >>> 16) & 0x8000, exponent = (x >>> 23) & 0xff, mantissa = x & 0x7fffff;
    if (exponent === 0xff) { out[i] = sign | (mantissa ? 0x7e00 : 0x7c00); continue; }
    let halfExp = exponent - 127 + 15;
    if (halfExp >= 31) out[i] = sign | 0x7c00;
    else if (halfExp <= 0) {
      if (halfExp < -10) out[i] = sign;
      else { const m = (mantissa | 0x800000) >>> (1 - halfExp); out[i] = sign | ((m + 0x1000) >>> 13); }
    } else out[i] = sign | (halfExp << 10) | ((mantissa + 0x1000) >>> 13);
  }
  return out;
}

function float16ToFloat32(values) {
  const out = new Float32Array(values.length), bits = new Uint32Array(out.buffer);
  for (let i = 0; i < values.length; i++) {
    const half = values[i], sign = (half & 0x8000) << 16, exponent = (half >>> 10) & 31, mantissa = half & 1023;
    if (!exponent) {
      if (!mantissa) bits[i] = sign;
      else { let m = mantissa, e = -14; while (!(m & 1024)) { m <<= 1; e--; } bits[i] = sign | ((e + 127) << 23) | ((m & 1023) << 13); }
    } else if (exponent === 31) bits[i] = sign | 0x7f800000 | (mantissa << 13);
    else bits[i] = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  return out;
}

async function tensorData(tensor) {
  const data = await tensor.getData();
  if (tensor.type !== 'float16') return data;
  return data instanceof Uint16Array ? float16ToFloat32(data) : Float32Array.from(data);
}

function detachBuffer(buffer) {
  if (!(buffer instanceof ArrayBuffer) || !buffer.byteLength) return 0;
  const bytes = buffer.byteLength;
  try {
    if (buffer.__soproShrinkable && typeof buffer.resize === 'function') buffer.resize(0);
    else if (typeof buffer.transfer === 'function') buffer.transfer(0);
    else if (typeof structuredClone === 'function') structuredClone(buffer, { transfer: [buffer] });
  } catch {}
  return buffer.byteLength === 0 ? bytes : 0;
}

function dispose(tensor) {
  if (!tensor) return;
  let buffer = null;
  try {
    if (tensor.location === 'cpu') {
      const data = tensor.data;
      if (ArrayBuffer.isView(data) && data.buffer instanceof ArrayBuffer && data.byteOffset === 0 && data.byteLength === data.buffer.byteLength) buffer = data.buffer;
    }
  } catch {}
  if (typeof tensor.dispose === 'function') tensor.dispose();
  detachBuffer(buffer);
}
function disposeAcousticState(state) {
  if (!state) return;
  for (const name of ['x', 'k', 'v']) {
    const value = state[name];
    if (Array.isArray(value)) value.forEach(dispose);
    else dispose(value);
  }
}
function disposeVocoderPromptState(prompt) {
  if (!prompt?.state) return;
  for (const name of ['embed', 'conv', 'pending0', 'pending1']) dispose(prompt.state[name]);
}

class Random {
  constructor(seed = 1) { this.state = (Number(seed) >>> 0) || 1; this.spare = null; }
  value() { let x = this.state; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.state = x >>> 0; return (this.state + 0.5) / 4294967296; }
  normal() { if (this.spare !== null) { const value = this.spare; this.spare = null; return value; } const r = Math.sqrt(-2 * Math.log(Math.max(1e-12, this.value()))), a = 2 * Math.PI * this.value(); this.spare = r * Math.sin(a); return r * Math.cos(a); }
}

function rotary(length, dim, offset = 0) {
  const cos = new Float32Array(length * dim), sin = new Float32Array(length * dim);
  for (let p = 0; p < length; p++) for (let j = 0; j < dim; j++) {
    const frequency = 1 / (10000 ** ((j % (dim / 2)) * 2 / dim)), angle = (p + offset) * frequency;
    cos[p * dim + j] = Math.cos(angle); sin[p * dim + j] = Math.sin(angle);
  }
  return { cos, sin };
}

function causalBias(length) {
  const bias = new Float32Array(length * length);
  for (let q = 0; q < length; q++) for (let k = q + 1; k < length; k++) bias[q * length + k] = -1e4;
  return bias;
}

function chunkBias(length, chunk) {
  const bias = new Float32Array(length * length);
  for (let q = 0; q < length; q++) { const end = Math.min(length, (Math.floor(q / chunk) + 1) * chunk); for (let k = end; k < length; k++) bias[q * length + k] = -1e4; }
  return bias;
}

function sample(logits, random, { temperature, topP, topK }, bos, eos, allowEos) {
  const n = logits.length;
  if (temperature <= 0) {
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < n; i++) { if (i === bos || (!allowEos && i === eos)) continue; if (logits[i] > bestScore) { bestScore = logits[i]; best = i; } }
    return best;
  }
  const limit = topK > 0 ? Math.min(topK, n) : n;
  const indices = new Int32Array(limit), scores = new Float64Array(limit);
  let count = 0, minScore = Infinity, minAt = 0;
  for (let i = 0; i < n; i++) {
    if (i === bos || (!allowEos && i === eos)) continue;
    const score = logits[i];
    if (count < limit) {
      indices[count] = i; scores[count] = score;
      if (score < minScore) { minScore = score; minAt = count; }
      count += 1;
      if (count === limit && limit < n) { minScore = Infinity; for (let j = 0; j < limit; j++) if (scores[j] < minScore) { minScore = scores[j]; minAt = j; } }
    } else if (score > minScore) {
      indices[minAt] = i; scores[minAt] = score;
      minScore = Infinity; for (let j = 0; j < limit; j++) if (scores[j] < minScore) { minScore = scores[j]; minAt = j; }
    }
  }
  const order = Array.from({ length: count }, (_, j) => j).sort((a, b) => scores[b] - scores[a]);
  const max = scores[order[0]] / temperature;
  let probabilities = order.map((j) => Math.exp(scores[j] / temperature - max));
  let total = probabilities.reduce((a, b) => a + b, 0);
  probabilities = probabilities.map((value) => value / total);
  if (topP < 1) {
    let cumulative = 0, keep = 0;
    do { cumulative += probabilities[keep++]; } while (keep < probabilities.length && cumulative < topP);
    order.length = keep; probabilities.length = keep; total = probabilities.reduce((a, b) => a + b, 0); probabilities = probabilities.map((value) => value / total);
  }
  let draw = random.value();
  for (let i = 0; i < probabilities.length; i++) { draw -= probabilities[i]; if (draw <= 0) return indices[order[i]]; }
  return indices[order.at(-1)];
}

function channelSlice(data, channels, frames, start, end) {
  const count = end - start, out = new Float32Array(channels * count);
  for (let c = 0; c < channels; c++) out.set(data.subarray(c * frames + start, c * frames + end), c * count);
  return out;
}

async function hashBuffer(data) { const digest = await crypto.subtle.digest('SHA-256', data); return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join(''); }

async function decodeMono(arrayBuffer, targetRate = 24000) {
  const Offline = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
  const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Offline && !Context) throw new Error('Web Audio is unavailable in this browser.');
  const context = Offline ? new Offline(1, 1, targetRate) : new Context({ sampleRate: targetRate });
  try {
    const audio = await context.decodeAudioData(arrayBuffer.slice(0));
    const mono = new Float32Array(audio.length);
    for (let channel = 0; channel < audio.numberOfChannels; channel++) { const source = audio.getChannelData(channel); for (let i = 0; i < mono.length; i++) mono[i] += source[i] / audio.numberOfChannels; }
    return audio.sampleRate === targetRate ? mono : resample(mono, audio.sampleRate, targetRate);
  } finally { if (context.close) await context.close(); }
}

class ModelStore {
  constructor(ort, baseUrl, manifest, profileName, cache, onProgress, lowMemory = false) {
    this.ort = ort; this.baseUrl = baseUrl; this.manifest = manifest; this.profileName = profileName; this.profile = manifest.profiles[profileName]; this.cache = cache; this.onProgress = onProgress; this.lowMemory = lowMemory; this.sessions = new Map();
    this.stagedShards = new Map();
    this.shardUses = new Map();
    for (const graph of Object.values(this.profile.graphs)) this.shardUses.set(graph.shard, (this.shardUses.get(graph.shard) ?? 0) + 1);
  }
  floatTensor(values, dims, dtype = this.profile.dtype) { const data = dtype === 'float16' ? float32ToFloat16(values) : (values instanceof Float32Array ? values : Float32Array.from(values)); return new this.ort.Tensor(dtype, data, dims); }
  graphFloatTensor(name, values, dims) { return this.floatTensor(values, dims, this.profile.graphs[name]?.dtype ?? this.profile.dtype); }
  boolTensor(value) { return new this.ort.Tensor('bool', Uint8Array.of(value ? 1 : 0), []); }
  intTensor(values, dims) { return new this.ort.Tensor('int64', int64(values), dims); }
  int32Tensor(values, dims) { return new this.ort.Tensor('int32', Int32Array.from(values), dims); }
  async releaseSession(name) {
    const pending = this.sessions.get(name);
    if (!pending) return;
    this.sessions.delete(name);
    try { await (await pending).release(); } catch {}
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  async releaseTransient(name) {
    if ((this.profile.graphs[name]?.provider ?? this.profile.provider) === 'webgpu') return;
    await this.releaseSession(name);
  }
  async releaseAll() { for (const name of [...this.sessions.keys()]) await this.releaseSession(name); this.releaseStagedWeights(); }
  releaseStagedWeights() {
    for (const [url, bytes] of this.stagedShards) {
      this.cache.release(url);
      detachBuffer(bytes.buffer);
    }
    this.stagedShards.clear();
  }
  releaseStagedShard(shardFile) {
    const file = fileName(shardFile);
    for (const [url, bytes] of [...this.stagedShards]) {
      if (fileName(url) !== file) continue;
      this.cache.release(url);
      detachBuffer(bytes.buffer);
      this.stagedShards.delete(url);
    }
  }
  async session(name) {
    if (this.sessions.has(name)) return this.sessions.get(name);
    const promise = this.#create(name); this.sessions.set(name, promise);
    try { return await promise; } catch (error) { this.sessions.delete(name); throw error; }
  }
  async preload(onProgress = this.onProgress) {
    const profileUrl = joinUrl(this.baseUrl, this.profileName), assets = new Map();
    for (const graph of Object.values(this.profile.graphs)) {
      assets.set(versionUrl(joinUrl(profileUrl, graph.url ?? graph.file), this.manifest.revision), graph.bytes ?? 0);
    }
    for (const shard of Object.values(this.profile.shards)) {
      assets.set(versionUrl(joinUrl(profileUrl, shard.url ?? shard.file), this.manifest.revision), shard.bytes ?? 0);
    }
    const total = [...assets.values()].reduce((sum, bytes) => sum + bytes, 0);
    let completed = 0;
    for (const [url, expected] of assets) {
      let current = 0;
      const loaded = await this.cache.prefetch(url, ({ loaded: value, total: assetTotal }) => {
        current = value;
        if (onProgress) onProgress({ url, loaded: completed + Math.min(value, expected || assetTotal || value), total });
      });
      completed += expected || loaded || current;
      if (onProgress) onProgress({ url, loaded: Math.min(completed, total || completed), total: total || completed });
    }
  }
  async #create(name) {
    const graph = this.profile.graphs[name]; if (!graph) throw new Error(`Graph ${name} is not present in ${this.profileName}.`);
    const profileUrl = joinUrl(this.baseUrl, this.profileName);
    const graphUrl = versionUrl(joinUrl(profileUrl, graph.url ?? graph.file), this.manifest.revision);
    const shard = Object.values(this.profile.shards).find((item) => item.file === graph.shard);
    const shardUrl = versionUrl(joinUrl(profileUrl, shard.url ?? shard.file), this.manifest.revision);
    const graphBytes = await this.cache.bytes(graphUrl, this.onProgress); this.cache.release(graphUrl);
    const sharedShard = (this.shardUses.get(graph.shard) ?? 0) > 1;
    const shardBytes = await this.cache.bytes(shardUrl, this.onProgress);
    if (sharedShard) this.stagedShards.set(shardUrl, shardBytes);
    else this.cache.release(shardUrl);
    const provider = graph.provider ?? this.profile.provider, gpuOutputs = {};
    if (provider === 'webgpu') {
      if (['semantic_prefill', 'semantic_step', 'semantic_stream', 'semantic_core'].includes(name)) Object.assign(gpuOutputs, { present_k: 'gpu-buffer', present_v: 'gpu-buffer' });
      if (name.startsWith('acoustic_stream_prefill_')) Object.assign(gpuOutputs, { x_context: 'gpu-buffer', present_k: 'gpu-buffer', present_v: 'gpu-buffer' });
      if (name.startsWith('acoustic_stream_step_')) Object.assign(gpuOutputs, { x_context_out: 'gpu-buffer', present_k: 'gpu-buffer', present_v: 'gpu-buffer' });
      if (name === 'vocoder_stream') Object.assign(gpuOutputs, { embed_state_out: 'gpu-buffer', conv_state_out: 'gpu-buffer', pending0_out: 'gpu-buffer', pending1_out: 'gpu-buffer' });
    }
    try {
      const session = await this.ort.InferenceSession.create(graphBytes, {
        executionProviders: [provider], graphOptimizationLevel: this.lowMemory ? 'basic' : 'all',
        externalData: [{ path: graph.shard, data: shardBytes }],
        ...(this.lowMemory ? { enableCpuMemArena: false, enableMemPattern: false, extra: { session: { disable_prepacking: '1' } } } : {}),
        ...(Object.keys(gpuOutputs).length ? { preferredOutputLocation: gpuOutputs } : {}),
      });
      return session;
    } finally {
      detachBuffer(graphBytes.buffer);
      if (!sharedShard) detachBuffer(shardBytes.buffer);
    }
  }
}

export class SoproTTS {
  static async create({ model = 'samuel-vitorino/sopro-v2-turbo-onnx', modelBaseUrl = null, revision = 'main', token = null, backend = 'auto', memory = 'auto', cache = true, wasmPaths = null, threads = null, onProgress = null } = {}) {
    if (!['auto', 'webgpu', 'wasm'].includes(backend)) throw new Error("backend must be 'auto', 'webgpu', or 'wasm'");
    if (!['auto', 'low', 'fast'].includes(memory)) throw new Error("memory must be 'auto', 'low', or 'fast'");
    if (threads !== null && (!Number.isInteger(threads) || threads < 1)) throw new Error('threads must be a positive integer');
    const base = modelBaseUrl ?? (/^(https?:)?\//.test(model) ? model : `https://huggingface.co/${model}/resolve/${revision}`);
    const headers = token ? { Authorization: `Bearer ${token}` } : null;
    const { mobileWebKit, mobile } = platform();
    if (backend === 'webgpu' && mobile) throw new Error('WebGPU is desktop-only. Use the WASM backend on mobile.');
    const persistentAssetCache = Boolean(cache) && !mobile;
    const bootstrap = new AssetCache('sopro-onnx-manifest-v1', persistentAssetCache, headers);
    const manifest = await bootstrap.json(joinUrl(base, 'manifest.json'), { revalidate: true });
    const assets = new AssetCache(`sopro-onnx-assets-${manifest.revision ?? manifest.model}`, persistentAssetCache, headers);
    let profile = 'wasm-uint8', adapter = null;
    if (backend !== 'wasm' && !mobile && globalThis.navigator?.gpu) {
      try {
        adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter?.features.has('shader-f16') && manifest.profiles['webgpu-fp16']) profile = 'webgpu-fp16';
        else if (adapter && manifest.profiles['webgpu-fp32']) profile = 'webgpu-fp32';
      } catch (error) {
        if (backend === 'webgpu') throw new Error('WebGPU adapter initialization failed.', { cause: error });
      }
    }
    if (backend === 'webgpu' && profile === 'wasm-uint8') throw new Error('WebGPU is unavailable.');
    if (!manifest.profiles[profile]) profile = Object.keys(manifest.profiles).find((name) => name.startsWith('wasm'));
    if (!profile) throw new Error('No compatible model profile is present.');
    const constrainedDevice = Number(navigator.deviceMemory || 0) > 0 && Number(navigator.deviceMemory) <= 6;
    const lowMemory = memory === 'low' || (memory === 'auto' && (mobile || constrainedDevice));
    const cores = Math.max(1, navigator.hardwareConcurrency || 4);
    const requestedThreads = typeof SharedArrayBuffer === 'undefined' ? 1 : (threads ?? (mobile
      ? Math.min(4, Math.ceil(cores / 2))
      : Math.min(4, cores)));
    const wasmProfile = profile.startsWith('wasm');
    const mobileThreadedWasm = wasmProfile && mobileWebKit && requestedThreads > 1;
    const zeroCopyWasm = wasmProfile && mobileWebKit;
    const ort = wasmProfile
      ? (zeroCopyWasm
        ? await import('./runtime/ort-mobile-loader.mjs?v=1')
        : await import('onnxruntime-web/wasm'))
      : await import('onnxruntime-web/webgpu');
    const mobileUnsharedWasm = wasmProfile && mobileWebKit && !mobileThreadedWasm;
    if (!wasmProfile) { ort.env.webgpu.adapter = adapter; ort.env.webgpu.powerPreference = 'high-performance'; }
    ort.env.wasm.numThreads = requestedThreads;
    ort.env.wasm.simd = true; ort.env.wasm.proxy = false;
    if (mobileUnsharedWasm) {
      ort.env.wasm.wasmPaths = {
        mjs: new URL('./runtime/ort-mobile.mjs?v=1', import.meta.url).href,
        wasm: new URL('./runtime/ort-mobile.wasm?v=1', import.meta.url).href,
      };
    } else if (mobileThreadedWasm) {
      ort.env.wasm.wasmPaths = {
        mjs: new URL('./runtime/ort-mobile-threaded.mjs?v=1', import.meta.url).href,
        wasm: new URL('./runtime/ort-mobile-threaded.wasm?v=1', import.meta.url).href,
      };
    } else if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
    const tokenizer = new SentencePieceTokenizer(await assets.json(versionUrl(joinUrl(base, manifest.tokenizer), manifest.revision)));
    const unifiedVocoder = Boolean(manifest.profiles[profile]?.graphs?.vocoder_stream);
    const runtimeVariant = mobileUnsharedWasm
      ? 'wasm-mobile'
      : mobileThreadedWasm ? `wasm-mobile-${requestedThreads}t` : null;
    return new SoproTTS(manifest, profile, tokenizer, new ModelStore(ort, base, manifest, profile, assets, onProgress, lowMemory), runtimeVariant);
  }

  constructor(manifest, profile, tokenizer, store, runtimeVariant = null) {
    this.manifest = manifest; this.backend = profile; this.tokenizer = tokenizer; this.store = store;
    this.runtimeVariant = runtimeVariant;
    this.sampleRate = manifest.sampleRate; this.config = manifest.config; this.hopRatio = manifest.hopRatio; this.tokenSamples = manifest.tokenSamples;
    this.promptStates = new Map();
    this.vocoderPromptStates = new Map();
  }

  async preload(onProgress = null) { await this.store.preload(onProgress); }

  async prepareReference(file, { seconds = null, sampleRate = null, cache = true } = {}) {
    const pcm = file instanceof Float32Array ? file : null;
    const buffer = pcm ? null : (file instanceof ArrayBuffer ? file : await file.arrayBuffer());
    const duration = seconds ?? this.config.generation.ref_seconds;
    const key = `${this.manifest.model}:${this.manifest.revision ?? 'unversioned'}:${duration}:${await hashBuffer(pcm ?? buffer)}`;
    if (cache) { const saved = await getCachedReference(key); if (saved) return { ...saved, fromCache: true }; }
    const decoded = pcm ? (sampleRate && sampleRate !== this.sampleRate ? resample(pcm, sampleRate, this.sampleRate) : pcm) : await decodeMono(buffer, this.sampleRate);
    const { wav, levelDb } = cropAndNormalizeReference(decoded, this.sampleRate, duration);
    const features = referenceFeatures(wav), session = await this.store.session('reference');
    const output = await session.run({
      semantic_mel: this.store.graphFloatTensor('reference', features.semantic.data, [1, 80, features.semantic.frames]),
      interp_left: this.store.intTensor(features.interp.left, [features.nTokens]),
      interp_right: this.store.intTensor(features.interp.right, [features.nTokens]),
      interp_weight: this.store.graphFloatTensor('reference', features.interp.weight, [features.nTokens]),
      speaker_mel: this.store.graphFloatTensor('reference', features.speaker.data, [1, 80, features.speaker.frames]),
    });
    const condVec = Float32Array.from(await tensorData(output.cond_vec));
    const semanticTokens = Uint32Array.from(await tensorData(output.semantic_tokens), Number);
    dispose(output.cond_vec); dispose(output.semantic_tokens);
    await this.store.releaseSession('reference');
    const mel = features.acoustic.data, frames = features.acoustic.frames, mean = this.config.model.acoustic_mel_mean, std = this.config.model.acoustic_mel_std;
    for (let m = 0; m < 100; m++) for (let t = 0; t < frames; t++) mel[m * frames + t] = (mel[m * frames + t] - mean[m]) / std[m];
    const reference = { key, condVec, semanticTokens, mel, melFrames: frames, levelDb };
    if (cache) await putCachedReference(key, reference);
    return reference;
  }

  async *_semanticStreamCore(text, reference, parameters, promptTokens = null) {
    const cfg = this.config, model = cfg.model, generation = cfg.generation;
    const textIds = this.tokenizer.encode(text, parameters.language), style = reference.semanticTokens.slice(0, generation.style_tokens), prompt = promptTokens ?? reference.semanticTokens.slice(0, generation.prompt_tokens);
    const prefixLength = model.style_prefix_tokens + textIds.length + prompt.length + 1, headDim = model.ar_model_dim / model.ar_heads;
    const prefix = await this.store.session('semantic_prefix'), embedded = await prefix.run({
      text_ids: this.store.intTensor(textIds, [1, textIds.length]), style_tokens: this.store.intTensor(style, [1, style.length]), prompt_tokens: this.store.intTensor(prompt, [1, prompt.length]),
    });
    dispose(embedded.bos_hidden);
    const core = await this.store.session('semantic_core'), emptyK = this.store.graphFloatTensor('semantic_core', new Float32Array(0), [model.ar_blocks, 1, model.ar_kv_heads, 0, headDim]), emptyV = this.store.graphFloatTensor('semantic_core', new Float32Array(0), [model.ar_blocks, 1, model.ar_kv_heads, 0, headDim]);
    const emptyHidden = this.store.graphFloatTensor('semantic_core', new Float32Array(0), [1, 0, model.ar_model_dim]), tokenBuffer = new BigInt64Array(1);
    tokenBuffer[0] = BigInt(model.semantic_vocab_size);
    let prefixHidden = embedded.hidden, output, pastK = null, pastV = null, logits, position = prefixLength;
    const random = new Random(parameters.seed), pending = [], maxSteps = Math.ceil(parameters.maxSeconds * this.sampleRate / this.tokenSamples), minSteps = Math.max(1, Math.ceil(parameters.minSeconds * this.sampleRate / this.tokenSamples));
    try {
      const rotation = rotary(prefixLength, headDim);
      output = await core.run({
        hidden: prefixHidden, token: this.store.intTensor(tokenBuffer, [1, 1]), past_k: emptyK, past_v: emptyV,
        cos: this.store.graphFloatTensor('semantic_core', rotation.cos, [prefixLength, headDim]), sin: this.store.graphFloatTensor('semantic_core', rotation.sin, [prefixLength, headDim]),
        attention_bias: this.store.graphFloatTensor('semantic_core', causalBias(prefixLength), [1, 1, prefixLength, prefixLength]),
      });
      dispose(prefixHidden); prefixHidden = null;
      logits = Float32Array.from(await tensorData(output.logits)); dispose(output.logits);
      pastK = output.present_k; pastV = output.present_v;
      for (let index = 0; index < maxSteps; index++) {
        const token = sample(logits, random, parameters, model.semantic_vocab_size, model.semantic_vocab_size + 1, index + 1 >= minSteps);
        if (index + 1 >= minSteps && token === model.semantic_vocab_size + 1) break;
        pending.push(Math.min(model.semantic_vocab_size - 1, Math.max(0, token)));
        if (pending.length >= parameters.semanticChunkTokens) yield pending.splice(0);
        if (index + 1 >= maxSteps) break;
        const r = rotary(1, headDim, position), previousK = pastK, previousV = pastV, presentLength = position + 1;
        position += 1; tokenBuffer[0] = BigInt(token);
        output = await core.run({
          hidden: emptyHidden, token: this.store.intTensor(tokenBuffer, [1, 1]), past_k: pastK, past_v: pastV,
          cos: this.store.graphFloatTensor('semantic_core', r.cos, [1, headDim]), sin: this.store.graphFloatTensor('semantic_core', r.sin, [1, headDim]),
          attention_bias: this.store.graphFloatTensor('semantic_core', new Float32Array(presentLength), [1, 1, 1, presentLength]),
        });
        logits = Float32Array.from(await tensorData(output.logits)); dispose(output.logits);
        pastK = output.present_k; pastV = output.present_v; dispose(previousK); dispose(previousV);
        await cooperate(parameters, index, 8);
      }
    } finally {
      dispose(prefixHidden); dispose(emptyK); dispose(emptyV); dispose(emptyHidden); dispose(pastK); dispose(pastV);
    }
    if (pending.length) yield pending;
  }

  async *_semanticStream(text, reference, parameters, promptTokens = null) {
    const graphs = this.manifest.profiles[this.backend]?.graphs ?? {};
    if (graphs.semantic_prefix && graphs.semantic_core) {
      yield* this._semanticStreamCore(text, reference, parameters, promptTokens);
      return;
    }
    const cfg = this.config, model = cfg.model, generation = cfg.generation;
    const textIds = this.tokenizer.encode(text, parameters.language), style = reference.semanticTokens.slice(0, generation.style_tokens), prompt = promptTokens ?? reference.semanticTokens.slice(0, generation.prompt_tokens);
    const prefixLength = model.style_prefix_tokens + textIds.length + prompt.length + 1, headDim = model.ar_model_dim / model.ar_heads;
    const unifiedSemantic = Boolean(this.manifest.profiles[this.backend]?.graphs?.semantic_stream), semanticName = unifiedSemantic ? 'semantic_stream' : 'semantic_prefill';
    const rotation = rotary(prefixLength, headDim), prefill = await this.store.session(semanticName);
    const semanticControls = unifiedSemantic ? {
      prefill: this.store.boolTensor(true), step: this.store.boolTensor(false),
      text_ids: this.store.intTensor(new BigInt64Array(0), [1, 0]), style_tokens: this.store.intTensor(new BigInt64Array(0), [1, 0]), prompt_tokens: this.store.intTensor(new BigInt64Array(0), [1, 0]),
      causal_bias: this.store.graphFloatTensor(semanticName, new Float32Array(0), [1, 1, 0, 0]), token: this.store.intTensor([0], [1, 1]),
      past_k: this.store.graphFloatTensor(semanticName, new Float32Array(0), [model.ar_blocks, 1, model.ar_kv_heads, 0, headDim]),
      past_v: this.store.graphFloatTensor(semanticName, new Float32Array(0), [model.ar_blocks, 1, model.ar_kv_heads, 0, headDim]),
    } : null;
    let output, pastK = null, pastV = null, logits, position = prefixLength;
    const random = new Random(parameters.seed), pending = [], maxSteps = Math.ceil(parameters.maxSeconds * this.sampleRate / this.tokenSamples), minSteps = Math.max(1, Math.ceil(parameters.minSeconds * this.sampleRate / this.tokenSamples));
    const tokenBuffer = new BigInt64Array(1);
    try {
      output = await prefill.run({
        text_ids: this.store.intTensor(textIds, [1, textIds.length]), style_tokens: this.store.intTensor(style, [1, style.length]), prompt_tokens: this.store.intTensor(prompt, [1, prompt.length]),
        cos: this.store.graphFloatTensor(semanticName, rotation.cos, [prefixLength, headDim]), sin: this.store.graphFloatTensor(semanticName, rotation.sin, [prefixLength, headDim]), causal_bias: this.store.graphFloatTensor(semanticName, causalBias(prefixLength), [1, 1, prefixLength, prefixLength]),
        ...(unifiedSemantic ? { is_prefill: semanticControls.prefill, token: semanticControls.token, past_k: semanticControls.past_k, past_v: semanticControls.past_v } : {}),
      });
      logits = Float32Array.from(await tensorData(output.logits)); dispose(output.logits);
      pastK = output.present_k; pastV = output.present_v;
      const stepName = unifiedSemantic ? semanticName : 'semantic_step', stepSession = await this.store.session(stepName);
      for (let index = 0; index < maxSteps; index++) {
        const token = sample(logits, random, parameters, model.semantic_vocab_size, model.semantic_vocab_size + 1, index + 1 >= minSteps);
        if (index + 1 >= minSteps && token === model.semantic_vocab_size + 1) break;
        pending.push(Math.min(model.semantic_vocab_size - 1, Math.max(0, token)));
        if (pending.length >= parameters.semanticChunkTokens) yield pending.splice(0);
        if (index + 1 >= maxSteps) break;
        const r = rotary(1, headDim, position++), previousK = pastK, previousV = pastV;
        tokenBuffer[0] = BigInt(token);
        output = await stepSession.run({
          token: this.store.intTensor(tokenBuffer, [1, 1]), past_k: pastK, past_v: pastV,
          cos: this.store.graphFloatTensor(stepName, r.cos, [1, headDim]), sin: this.store.graphFloatTensor(stepName, r.sin, [1, headDim]),
          ...(unifiedSemantic ? { is_prefill: semanticControls.step, text_ids: semanticControls.text_ids, style_tokens: semanticControls.style_tokens, prompt_tokens: semanticControls.prompt_tokens, causal_bias: semanticControls.causal_bias } : {}),
        });
        logits = Float32Array.from(await tensorData(output.logits)); dispose(output.logits);
        pastK = output.present_k; pastV = output.present_v; dispose(previousK); dispose(previousV);
        await cooperate(parameters, index, 8);
      }
    } finally {
      dispose(pastK); dispose(pastV);
      if (semanticControls) Object.values(semanticControls).forEach(dispose);
    }
    if (pending.length) yield pending;
  }

  _parameters(options = {}) {
    const defaults = this.config.generation, steps = Number(options.steps ?? defaults.steps);
    if (steps !== this.manifest.steps) throw new Error(`steps is fixed at ${this.manifest.steps}`);
    return { language: options.language ?? null, temperature: Number(options.temperature ?? defaults.temperature), topP: Number(options.topP ?? defaults.top_p), topK: Number(options.topK ?? defaults.top_k), steps, maxSeconds: Number(options.maxSeconds ?? (this.store.lowMemory ? Math.min(15, defaults.max_seconds) : defaults.max_seconds)), minSeconds: Number(options.minSeconds ?? defaults.min_seconds), seed: Number(options.seed ?? 1), semanticChunkTokens: Number(options.semanticChunkTokens ?? defaults.stream_chunk_frames / this.hopRatio), maxSegmentChars: Number(options.maxSegmentChars ?? (this.store.lowMemory ? 140 : defaults.max_segment_chars)), cooperative: Boolean(options.cooperative) };
  }

  _usesSplitAcoustic(parameters) {
    const graphs = this.manifest.profiles[this.backend]?.graphs ?? {};
    return this.backend.startsWith('wasm') && Array.from({ length: parameters.steps }, (_, step) =>
      graphs[`acoustic_stream_prefill_ode_${parameters.steps}_${step}`] && graphs[`acoustic_stream_ode_${parameters.steps}_${step}`]).every(Boolean);
  }

  async _condition(tokens, totalFrames, start = 0, end = totalFrames, transient = false, cache = null) {
    if (!cache || start > cache.upTo) return this._conditionWindow(tokens, totalFrames, start, end, transient);
    const compute = Math.max(start, cache.upTo);
    if (compute < end) {
      const fresh = await this._conditionWindow(tokens, totalFrames, compute, end, transient);
      const need = end * 100;
      if (cache.data.length < need) {
        const grown = new Float32Array(Math.max(need, cache.data.length * 2));
        grown.set(cache.data);
        cache.data = grown;
      }
      for (let m = 0; m < 100; m++) for (let f = compute; f < end; f++) cache.data[f * 100 + m] = fresh[m * (end - compute) + (f - compute)];
      cache.upTo = end;
    }
    const count = end - start, out = new Float32Array(100 * count);
    for (let m = 0; m < 100; m++) for (let f = start; f < end; f++) out[m * count + (f - start)] = cache.data[f * 100 + m];
    return out;
  }

  async _conditionWindow(tokens, totalFrames, start, end, transient) {
    const leftContext = (this.config.model.acoustic_upsampler_kernel_size ?? 3) - 1, first = Math.max(0, start - leftContext);
    const CONDITION_TOKEN_MARGIN = 16;
    const frameToken = (i) => Math.min(tokens.length - 1, Math.floor(i * tokens.length / totalFrames));
    const sliceStart = Math.max(0, frameToken(first) - CONDITION_TOKEN_MARGIN);
    const sliceEnd = Math.min(tokens.length, frameToken(end - 1) + 1 + CONDITION_TOKEN_MARGIN);
    const slice = tokens.slice(sliceStart, sliceEnd);
    const map = new BigInt64Array(end - first); for (let i = first; i < end; i++) map[i - first] = BigInt(frameToken(i) - sliceStart);
    const session = await this.store.session('acoustic_condition');
    const output = await session.run({ semantic_tokens: this.store.intTensor(slice, [1, slice.length]), frame_to_token: this.store.intTensor(map, [map.length]) });
    const mu = Float32Array.from(await tensorData(output.mu)), frames = end - first;
    dispose(output.mu);
    if (transient) await this.store.releaseTransient('acoustic_condition');
    return first === start ? mu : channelSlice(mu, 100, frames, start - first, frames);
  }

  _muCache(promptState) {
    const frames = promptState.muFrames ?? 0, data = new Float32Array((frames + 512) * 100);
    if (frames) for (let m = 0; m < 100; m++) for (let f = 0; f < frames; f++) data[f * 100 + m] = promptState.mu[m * frames + f];
    return { data, upTo: frames };
  }

  async synthesize(text, reference, options = {}) {
    const parameters = this._parameters(options), generation = this.config.generation;
    const parts = [];
    let carry = null;
    for (const segment of splitText(text, parameters.maxSegmentChars)) {
      const { wav, tokens } = await this._synthesizeSegment(segment, reference, parameters, carry);
      if (tokens.length && generation.prompt_tokens > 0) carry = tokens.slice(-generation.prompt_tokens);
      if (wav.length) parts.push(wav);
    }
    if (!parts.length) return new Float32Array(this.tokenSamples);
    const concat = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
    let offset = 0; for (const part of parts) { concat.set(part, offset); offset += part.length; }
    const gain = matchGain(concat, this.sampleRate);
    const trimmed = parts.map((part, index) => {
      const scaled = new Float32Array(part.length);
      for (let i = 0; i < part.length; i++) scaled[i] = part[i] * gain;
      const led = index === 0 ? trimLead(scaled, this.sampleRate) : trimLead(scaled, this.sampleRate, SEGMENT_LEAD_SECONDS, SEGMENT_SKIP_SECONDS);
      return trimTrail(led, this.sampleRate);
    });
    return fadeEdges(softLimit(joinSegments(trimmed, this.sampleRate)), this.sampleRate, false, true, FINAL_FADE_SECONDS);
  }

  async _synthesizeSegment(text, reference, parameters, promptTokens = null) {
    const generated = [];
    for await (const chunk of this._semanticStream(text, reference, parameters, promptTokens)) generated.push(...chunk);
    if (!generated.length) return { wav: new Float32Array(0), tokens: new Uint32Array(0) };
    const tokens = Uint32Array.from(generated);
    const allTokens = Uint32Array.from([...reference.semanticTokens, ...generated]), frames = reference.melFrames + generated.length * this.hopRatio;
    const mu = await this._condition(allTokens, frames, 0, frames, this.store.lowMemory), x0 = new Float32Array(100 * frames), random = new Random(parameters.seed ^ 0xa53c9e1d);
    for (let i = 0; i < x0.length; i++) x0[i] = random.normal();
    const condMel = new Float32Array(100 * frames), condMask = new Float32Array(frames);
    condMask.fill(1, 0, reference.melFrames);
    for (let m = 0; m < 100; m++) condMel.set(reference.mel.subarray(m * reference.melFrames, (m + 1) * reference.melFrames), m * frames);
    const rotation = rotary(frames, this.config.model.acoustic_dit_dim_head), acoustic = await this.store.session(`acoustic_offline_${parameters.steps}`);
    const output = await acoustic.run({ x_init: this.store.graphFloatTensor(`acoustic_offline_${parameters.steps}`, x0, [1, 100, frames]), mu: this.store.graphFloatTensor(`acoustic_offline_${parameters.steps}`, mu, [1, 100, frames]), cond_vec: this.store.graphFloatTensor(`acoustic_offline_${parameters.steps}`, reference.condVec, [1, reference.condVec.length]), cond_mel: this.store.graphFloatTensor(`acoustic_offline_${parameters.steps}`, condMel, [1, 100, frames]), cond_mask: this.store.graphFloatTensor(`acoustic_offline_${parameters.steps}`, condMask, [1, 1, frames]), cos: this.store.graphFloatTensor(`acoustic_offline_${parameters.steps}`, rotation.cos, [frames, rotation.cos.length / frames]), sin: this.store.graphFloatTensor(`acoustic_offline_${parameters.steps}`, rotation.sin, [frames, rotation.sin.length / frames]) });
    const normalized = Float32Array.from(await tensorData(output.mel)), context = Math.min(32, reference.melFrames), decodeFrames = context + generated.length * this.hopRatio;
    dispose(output.mel);
    if (this.store.lowMemory) await this.store.releaseTransient(`acoustic_offline_${parameters.steps}`);
    const mel = channelSlice(normalized, 100, frames, reference.melFrames - context, frames), mean = this.config.model.acoustic_mel_mean, std = this.config.model.acoustic_mel_std;
    for (let m = 0; m < 100; m++) for (let t = 0; t < decodeFrames; t++) mel[m * decodeFrames + t] = mel[m * decodeFrames + t] * std[m] + mean[m];
    const vocoder = await this.store.session('vocoder_offline'), decoded = await vocoder.run({ mel: this.store.graphFloatTensor('vocoder_offline', mel, [1, 100, decodeFrames]) });
    const featureFrames = decoded.istft_features.dims[1], features = Float32Array.from(await tensorData(decoded.istft_features));
    dispose(decoded.istft_features);
    if (this.store.lowMemory) await this.store.releaseTransient('vocoder_offline');
    const waveform = istft(features, featureFrames);
    const start = context * this.config.vocoder.hop_length, target = generated.length * this.tokenSamples;
    return { wav: waveform.slice(start, Math.min(waveform.length, start + target)), tokens };
  }

  async _promptState(reference, parameters) {
    const chunkFrames = this.config.generation.stream_chunk_frames, pMel = reference.melFrames;
    const stablePrompt = Math.floor(Math.min(pMel, Math.max(0, reference.semanticTokens.length - this.config.model.acoustic_pre_lookahead_frames) * this.hopRatio) / chunkFrames) * chunkFrames;
    if (stablePrompt < this.manifest.positionContext) return null;
    const promptFrames = stablePrompt;
    const key = `${reference.key}:${parameters.seed}:${parameters.steps}:${promptFrames}`;
    if (this.promptStates.has(key)) return this.promptStates.get(key);
    const promise = (async () => {
      const random = new Random(parameters.seed ^ 0x71d3a4c9), noise = new Float32Array(100 * promptFrames);
      for (let m = 0; m < 100; m++) for (let t = 0; t < pMel; t++) { const value = random.normal(); if (t < promptFrames) noise[m * promptFrames + t] = value; }
      const mu = await this._condition(reference.semanticTokens, pMel, 0, promptFrames), rotation = rotary(promptFrames, this.config.model.acoustic_dit_dim_head);
      if (this._usesSplitAcoustic(parameters)) {
        const grid = this.manifest.acousticGrid;
        if (!Array.isArray(grid) || grid.length !== parameters.steps + 1) throw new Error('The model manifest is missing its acoustic solver grid.');
        const firstName = `acoustic_stream_prefill_ode_${parameters.steps}_0`, xInit = this.store.graphFloatTensor(firstName, noise, [1, 100, promptFrames]);
        let x = xInit;
        const contexts = [], keys = [], values = [];
        try {
          for (let step = 0; step < parameters.steps; step++) {
            const name = `acoustic_stream_prefill_ode_${parameters.steps}_${step}`, prefill = await this.store.session(name), previous = x;
            const output = await prefill.run({
              x, x_init: xInit,
              mu: this.store.graphFloatTensor(name, mu, [1, 100, promptFrames]), cond_vec: this.store.graphFloatTensor(name, reference.condVec, [1, reference.condVec.length]),
              cond_mel: this.store.graphFloatTensor(name, channelSlice(reference.mel, 100, pMel, 0, promptFrames), [1, 100, promptFrames]), cond_mask: this.store.graphFloatTensor(name, new Float32Array(promptFrames).fill(1), [1, 1, promptFrames]),
              cos: this.store.graphFloatTensor(name, rotation.cos, [promptFrames, this.config.model.acoustic_dit_dim_head]), sin: this.store.graphFloatTensor(name, rotation.sin, [promptFrames, this.config.model.acoustic_dit_dim_head]),
              chunk_bias: this.store.graphFloatTensor(name, chunkBias(promptFrames, chunkFrames), [1, 1, promptFrames, promptFrames]),
            }, ['x_out', 'x_context_out', 'present_k', 'present_v']);
            x = output.x_out; contexts.push(output.x_context_out); keys.push(output.present_k); values.push(output.present_v);
            if (previous !== xInit) dispose(previous);
            await this.store.releaseTransient(name);
          }
          return { cached: promptFrames, x: contexts, k: keys, v: values, splitOde: true, mu, muFrames: promptFrames };
        } catch (error) {
          contexts.forEach(dispose); keys.forEach(dispose); values.forEach(dispose);
          throw error;
        } finally {
          if (x !== xInit) dispose(x);
          dispose(xInit);
        }
      }
      const name = `acoustic_stream_prefill_${parameters.steps}`, prefill = await this.store.session(name);
      const output = await prefill.run({
        x_init: this.store.graphFloatTensor(name, noise, [1, 100, promptFrames]), mu: this.store.graphFloatTensor(name, mu, [1, 100, promptFrames]), cond_vec: this.store.graphFloatTensor(name, reference.condVec, [1, reference.condVec.length]), cond_mel: this.store.graphFloatTensor(name, channelSlice(reference.mel, 100, pMel, 0, promptFrames), [1, 100, promptFrames]), cond_mask: this.store.graphFloatTensor(name, new Float32Array(promptFrames).fill(1), [1, 1, promptFrames]), cos: this.store.graphFloatTensor(name, rotation.cos, [promptFrames, this.config.model.acoustic_dit_dim_head]), sin: this.store.graphFloatTensor(name, rotation.sin, [promptFrames, this.config.model.acoustic_dit_dim_head]), chunk_bias: this.store.graphFloatTensor(name, chunkBias(promptFrames, chunkFrames), [1, 1, promptFrames, promptFrames]),
      });
      await this.store.releaseTransient(name);
      return { cached: promptFrames, x: output.x_context, k: output.present_k, v: output.present_v, mu, muFrames: promptFrames };
    })();
    for (const [staleKey, stale] of this.promptStates) {
      this.promptStates.delete(staleKey);
      stale.then(disposeAcousticState).catch(() => {});
    }
    this.promptStates.set(key, promise);
    try { return await promise; } catch (error) { this.promptStates.delete(key); throw error; }
  }

  async _vocoderPromptState(reference) {
    const name = 'vocoder_stream';
    const warmFrames = Math.min(32, reference.melFrames), key = `${reference.key}:${this.backend}:${warmFrames}`;
    if (this.vocoderPromptStates.has(key)) return this.vocoderPromptStates.get(key);
    const promise = (async () => {
      const warm = channelSlice(reference.mel, 100, reference.melFrames, reference.melFrames - warmFrames, reference.melFrames);
      const mean = this.config.model.acoustic_mel_mean, std = this.config.model.acoustic_mel_std;
      for (let m = 0; m < 100; m++) for (let t = 0; t < warmFrames; t++) warm[m * warmFrames + t] = warm[m * warmFrames + t] * std[m] + mean[m];
      const controls = {
        start: this.store.boolTensor(true), flush: this.store.boolTensor(false),
        embed: this.store.graphFloatTensor(name, new Float32Array(this.config.vocoder_streaming.n_mels * 6), [1, this.config.vocoder_streaming.n_mels, 6]),
        conv: this.store.graphFloatTensor(name, new Float32Array(this.config.vocoder_streaming.num_layers * this.config.vocoder_streaming.dim * 6), [this.config.vocoder_streaming.num_layers, 1, this.config.vocoder_streaming.dim, 6]),
        pending0: this.store.graphFloatTensor(name, new Float32Array(this.config.vocoder_streaming.dim * (this.config.vocoder_streaming.block_lookaheads?.[0] ?? 0)), [1, this.config.vocoder_streaming.dim, this.config.vocoder_streaming.block_lookaheads?.[0] ?? 0]),
        pending1: this.store.graphFloatTensor(name, new Float32Array(this.config.vocoder_streaming.dim * (this.config.vocoder_streaming.block_lookaheads?.[1] ?? 0)), [1, this.config.vocoder_streaming.dim, this.config.vocoder_streaming.block_lookaheads?.[1] ?? 0]),
      };
      let output;
      try {
        output = await (await this.store.session(name)).run({
          mel: this.store.graphFloatTensor(name, warm, [1, 100, warmFrames]),
          is_start: controls.start, is_flush: controls.flush,
          embed_state: controls.embed, conv_state: controls.conv,
          pending0: controls.pending0, pending1: controls.pending1,
        });
        const featureFrames = output.istft_features.dims[1], features = Float32Array.from(await tensorData(output.istft_features));
        dispose(output.istft_features);
        return {
          warmFrames, featureFrames, features,
          state: { embed: output.embed_state_out, conv: output.conv_state_out, pending0: output.pending0_out, pending1: output.pending1_out, shared: true },
        };
      } catch (error) {
        if (output) for (const tensor of Object.values(output)) dispose(tensor);
        throw error;
      } finally {
        Object.values(controls).forEach(dispose);
      }
    })();
    for (const [staleKey, stale] of this.vocoderPromptStates) {
      this.vocoderPromptStates.delete(staleKey);
      stale.then(disposeVocoderPromptState).catch(() => {});
    }
    this.vocoderPromptStates.set(key, promise);
    try { return await promise; } catch (error) { this.vocoderPromptStates.delete(key); throw error; }
  }

  async prepareStreaming(reference, options = {}) {
    const parameters = this._parameters(options);
    await this._promptState(reference, parameters);
    await cooperate(parameters);
    const acousticSteps = this._usesSplitAcoustic(parameters)
      ? Array.from({ length: parameters.steps }, (_, step) => `acoustic_stream_ode_${parameters.steps}_${step}`)
      : [`acoustic_stream_step_${parameters.steps}`];
    const profileGraphs = this.manifest.profiles[this.backend]?.graphs ?? {};
    const semanticGraphs = profileGraphs.semantic_prefix && profileGraphs.semantic_core ? ['semantic_prefix', 'semantic_core'] : (profileGraphs.semantic_stream ? ['semantic_stream'] : ['semantic_prefill', 'semantic_step']);
    for (const name of acousticSteps) {
      await this.store.session(name);
      await cooperate(parameters);
    }
    this.store.releaseStagedShard(profileGraphs[acousticSteps[0]].shard);
    for (const name of semanticGraphs) {
      await this.store.session(name);
      await cooperate(parameters);
    }
    this.store.releaseStagedShard(profileGraphs[semanticGraphs[0]].shard);
    await this.store.session('acoustic_condition');
    await cooperate(parameters);
    await this._vocoderPromptState(reference);
    await cooperate(parameters);
    await this.store.session('vocoder_stream');
    this.store.releaseStagedWeights();
    if (this.backend.startsWith('wasm')) {
      await this._warmup(parameters, acousticSteps, semanticGraphs);
      await cooperate(parameters);
    }
  }

  async _warmup(parameters, acousticSteps, semanticGraphs) {
    try {
      const model = this.config.model, headDim = model.ar_model_dim / model.ar_heads, dim = model.acoustic_dit_dim_head, ctx = this.manifest.positionContext, count = this.config.generation.stream_chunk_frames;
      if (semanticGraphs.includes('semantic_core')) {
        const fill = (n) => Array.from({ length: n }, (_, i) => (i % 96) + 1);
        const prefix = await this.store.session('semantic_prefix');
        const embedded = await prefix.run({ text_ids: this.store.intTensor(fill(48), [1, 48]), style_tokens: this.store.intTensor(fill(this.config.generation.style_tokens), [1, this.config.generation.style_tokens]), prompt_tokens: this.store.intTensor(fill(this.config.generation.prompt_tokens), [1, this.config.generation.prompt_tokens]) });
        const length = embedded.hidden.dims[1] + 1, rotation = rotary(length, headDim);
        const core = await this.store.session('semantic_core');
        const prefill = await core.run({
          hidden: embedded.hidden, token: this.store.intTensor([1], [1, 1]),
          past_k: this.store.graphFloatTensor('semantic_core', new Float32Array(0), [model.ar_blocks, 1, model.ar_kv_heads, 0, headDim]),
          past_v: this.store.graphFloatTensor('semantic_core', new Float32Array(0), [model.ar_blocks, 1, model.ar_kv_heads, 0, headDim]),
          cos: this.store.graphFloatTensor('semantic_core', rotation.cos, [length, headDim]), sin: this.store.graphFloatTensor('semantic_core', rotation.sin, [length, headDim]),
          attention_bias: this.store.graphFloatTensor('semantic_core', causalBias(length), [1, 1, length, length]),
        });
        let pastK = prefill.present_k, pastV = prefill.present_v;
        for (let i = 0; i < 8; i++) {
          const r = rotary(1, headDim, length + i);
          const step = await core.run({
            hidden: this.store.graphFloatTensor('semantic_core', new Float32Array(0), [1, 0, model.ar_model_dim]), token: this.store.intTensor([1], [1, 1]),
            past_k: pastK, past_v: pastV,
            cos: this.store.graphFloatTensor('semantic_core', r.cos, [1, headDim]), sin: this.store.graphFloatTensor('semantic_core', r.sin, [1, headDim]),
            attention_bias: this.store.graphFloatTensor('semantic_core', new Float32Array(length + 1 + i), [1, 1, 1, length + 1 + i]),
          });
          dispose(pastK); dispose(pastV); dispose(step.logits);
          pastK = step.present_k; pastV = step.present_v;
        }
        dispose(pastK); dispose(pastV);
        dispose(embedded.hidden); dispose(embedded.bos_hidden); dispose(prefill.logits);
      }
      const conditionSession = await this.store.session('acoustic_condition');
      const conditionOut = await conditionSession.run({ semantic_tokens: this.store.intTensor([1, 2, 3, 4, 5, 6, 7, 8], [1, 8]), frame_to_token: this.store.intTensor([0, 1, 2, 3, 4, 5, 6, 7], [8]) });
      Object.values(conditionOut).forEach(dispose);
      const r = rotary(count, dim), window = ctx + count, warmPast = 1024;
      let past = { k: this.store.floatTensor(new Float32Array(model.acoustic_dit_depth * model.acoustic_dit_heads * warmPast * dim), [model.acoustic_dit_depth, 1, model.acoustic_dit_heads, warmPast, dim], 'float16'), v: null, x: this.store.graphFloatTensor(acousticSteps[0], new Float32Array(100 * ctx), [1, 100, ctx]) };
      past.v = this.store.floatTensor(new Float32Array(model.acoustic_dit_depth * model.acoustic_dit_heads * warmPast * dim), [model.acoustic_dit_depth, 1, model.acoustic_dit_heads, warmPast, dim], 'float16');
      for (const name of acousticSteps) {
        const session = await this.store.session(name), zeros = (n) => new Float32Array(n);
        const feeds = {
          x_init: this.store.graphFloatTensor(name, zeros(100 * count), [1, 100, count]),
          mu_window: this.store.graphFloatTensor(name, zeros(100 * window), [1, 100, window]),
          cond_vec: this.store.graphFloatTensor(name, zeros(this.config.model.cond_hidden_dim), [1, this.config.model.cond_hidden_dim]),
          cond_mel_window: this.store.graphFloatTensor(name, zeros(100 * window), [1, 100, window]),
          cond_mask_window: this.store.graphFloatTensor(name, zeros(window), [1, 1, window]),
          cos: this.store.graphFloatTensor(name, r.cos, [count, dim]), sin: this.store.graphFloatTensor(name, r.sin, [count, dim]),
          x_context: past.x, past_k: past.k, past_v: past.v,
        };
        if (name.includes('_ode_')) feeds.x = feeds.x_init;
        const out = await session.run(feeds);
        Object.values(out).forEach(dispose);
      }
      dispose(past.k); dispose(past.v); dispose(past.x);
    } catch {}
  }

  async *stream(text, reference, options = {}) {
    const parameters = this._parameters(options), generation = this.config.generation;
    const promptState = await this._promptState(reference, parameters);
    if (!promptState) { yield await this.synthesize(text, reference, options); return; }
    const segments = splitText(text, parameters.maxSegmentChars);
    let carry = null;
    for (let index = 0; index < segments.length; index++) {
      const tokens = yield* this._streamSegment(segments[index], reference, parameters, promptState, index === 0, index + 1 === segments.length, carry);
      if (tokens.length && generation.prompt_tokens > 0) carry = tokens.slice(-generation.prompt_tokens);
    }
  }

  async *_streamSegment(text, reference, parameters, promptState, first, last, promptTokens) {
    const chunkFrames = this.config.generation.stream_chunk_frames, pMel = reference.melFrames, gain = outputGain(reference.levelDb);
    const noise = Array.from({ length: 100 }, () => []), random = new Random(parameters.seed ^ 0x71d3a4c9);
    const ensureNoise = (length) => { for (const channel of noise) while (channel.length < length) channel.push(random.normal()); };
    const noiseSlice = (start, end) => { ensureNoise(end); const out = new Float32Array(100 * (end - start)); for (let m = 0; m < 100; m++) out.set(noise[m].slice(start, end), m * (end - start)); return out; };
    ensureNoise(pMel);
    let acousticState = { ...promptState, shared: true };
    const muCache = this._muCache(promptState);
    const vocoderPrompt = await this._vocoderPromptState(reference), streamIstft = new StreamingISTFT(), warmFrames = vocoderPrompt.warmFrames, mean = this.config.model.acoustic_mel_mean, std = this.config.model.acoustic_mel_std;
    const warmAudio = streamIstft.push(vocoderPrompt.features, vocoderPrompt.featureFrames, false);
    const vocoderName = 'vocoder_stream', vocoderControls = {
      continue: this.store.boolTensor(false), flush: this.store.boolTensor(true),
    };
    let vocoderState = { ...vocoderPrompt.state, shared: true }, skipSamples = Math.max(0, warmFrames * this.config.vocoder_streaming.hop_length - warmAudio.length), emitted = 0;
    const lead = first ? LEAD_IN_SECONDS : SEGMENT_LEAD_SECONDS, skip = first ? 0 : SEGMENT_SKIP_SECONDS;
    let pending = new Float32Array(0), pendingOffset = 0;
    const gate = (audio) => {
      if (pending === null) return audio;
      const merged = new Float32Array(pending.length + audio.length);
      merged.set(pending); merged.set(audio, pending.length);
      const onset = speechOnset(merged, this.sampleRate);
      if (onset === null) {
        const keep = Math.floor(Math.max(lead, GATE_HOLD_SECONDS) * this.sampleRate);
        if (merged.length > keep) { pendingOffset += merged.length - keep; pending = merged.slice(merged.length - keep); }
        else pending = merged;
        return null;
      }
      const absoluteOnset = pendingOffset + onset;
      let cut = Math.max(absoluteOnset - Math.floor(lead * this.sampleRate), Math.floor(skip * this.sampleRate));
      cut = Math.min(cut, Math.max(0, absoluteOnset - Math.floor(0.02 * this.sampleRate)));
      const guard = energyOnset(merged, this.sampleRate);
      if (guard !== null && guard < onset) {
        const near = onset - guard <= Math.floor(0.05 * this.sampleRate);
        if (near || energyFraction(merged, this.sampleRate, guard, onset) >= 0.35) {
          cut = Math.min(cut, Math.max(Math.floor(skip * this.sampleRate), pendingOffset + guard - Math.floor(0.05 * this.sampleRate)));
        }
      }
      const out = merged.slice(Math.max(0, cut - pendingOffset));
      pending = null;
      return fadeEdges(out, this.sampleRate, !first, false);
    };
    const feedVocoder = async (mel, frames) => {
      const session = await this.store.session(vocoderName);
      const feeds = {
        mel: this.store.graphFloatTensor(vocoderName, mel, [1, 100, frames]),
        is_start: vocoderControls.continue, is_flush: vocoderControls.continue,
        embed_state: vocoderState.embed, conv_state: vocoderState.conv,
        pending0: vocoderState.pending0, pending1: vocoderState.pending1,
      };
      const old = vocoderState, out = await session.run(feeds); vocoderState = { embed: out.embed_state_out, conv: out.conv_state_out, pending0: out.pending0_out, pending1: out.pending1_out };
      if (old && !old.shared) for (const name of ['embed', 'conv', 'pending0', 'pending1']) dispose(old[name]);
      const featureFrames = out.istft_features.dims[1], features = Float32Array.from(await tensorData(out.istft_features));
      dispose(out.istft_features);
      let audio = streamIstft.push(features, featureFrames, false);
      if (skipSamples) { const drop = Math.min(skipSamples, audio.length); audio = audio.slice(drop); skipSamples -= drop; }
      return audio;
    };
    const generated = [];
    try {
    const splitOde = Boolean(promptState.splitOde);
    const stepNames = splitOde
      ? Array.from({ length: parameters.steps }, (_, step) => `acoustic_stream_ode_${parameters.steps}_${step}`)
      : [`acoustic_stream_step_${parameters.steps}`];
    const stepSessions = await Promise.all(stepNames.map((name) => this.store.session(name)));
    const advance = async (final) => {
      const chunks = [], allTokens = Uint32Array.from([...reference.semanticTokens, ...generated]), total = pMel + generated.length * this.hopRatio;
      let target = final ? total : Math.floor(Math.max(0, total - (this.config.model.acoustic_pos_kernel_size - 1 + this.hopRatio)) / chunkFrames) * chunkFrames;
      while (acousticState.cached < target) {
        const start = acousticState.cached, count = Math.min(chunkFrames, target - start), end = start + count;
        const windowStart = start - this.manifest.positionContext, muWindow = await this._condition(allTokens, total, windowStart, end, false, muCache);
        const condWindow = new Float32Array(100 * (this.manifest.positionContext + count)), maskWindow = new Float32Array(this.manifest.positionContext + count);
        for (let m = 0; m < 100; m++) for (let p = windowStart; p < end; p++) if (p < pMel) condWindow[m * (this.manifest.positionContext + count) + p - windowStart] = reference.mel[m * pMel + p];
        for (let p = windowStart; p < end; p++) if (p < pMel) maskWindow[p - windowStart] = 1;
        const r = rotary(count, this.config.model.acoustic_dit_dim_head, start), old = acousticState;
        let normalized;
        if (splitOde) {
          const noiseChunk = noiseSlice(start, end), xInit = this.store.graphFloatTensor(stepNames[0], noiseChunk, [1, 100, count]);
          let x = xInit;
          const next = { cached: end, x: [], k: [], v: [], shared: false, splitOde: true };
          try {
            for (let step = 0; step < parameters.steps; step++) {
              const previous = x, finalStep = step + 1 === parameters.steps, odeName = stepNames[step];
              const out = await stepSessions[step].run({
                x, x_init: xInit,
                mu_window: this.store.graphFloatTensor(odeName, muWindow, [1, 100, this.manifest.positionContext + count]), cond_vec: this.store.graphFloatTensor(odeName, reference.condVec, [1, reference.condVec.length]),
                cond_mel_window: this.store.graphFloatTensor(odeName, condWindow, [1, 100, this.manifest.positionContext + count]), cond_mask_window: this.store.graphFloatTensor(odeName, maskWindow, [1, 1, this.manifest.positionContext + count]),
                cos: this.store.graphFloatTensor(odeName, r.cos, [count, this.config.model.acoustic_dit_dim_head]), sin: this.store.graphFloatTensor(odeName, r.sin, [count, this.config.model.acoustic_dit_dim_head]),
                x_context: old.x[step], past_k: old.k[step], past_v: old.v[step],
              }, finalStep ? ['x_out', 'mel', 'x_context_out', 'present_k', 'present_v'] : ['x_out', 'x_context_out', 'present_k', 'present_v']);
              x = out.x_out; next.x.push(out.x_context_out); next.k.push(out.present_k); next.v.push(out.present_v);
              if (!old.shared) { dispose(old.x[step]); dispose(old.k[step]); dispose(old.v[step]); }
              if (finalStep) { normalized = Float32Array.from(await tensorData(out.mel)); dispose(out.mel); }
              if (previous !== xInit) dispose(previous);
            }
            acousticState = next;
          } catch (error) {
            disposeAcousticState(next);
            throw error;
          } finally {
            if (x !== xInit) dispose(x);
            dispose(xInit);
          }
        } else {
          const stepName = stepNames[0];
          const out = await stepSessions[0].run({
            x_init: this.store.graphFloatTensor(stepName, noiseSlice(start, end), [1, 100, count]),
            mu_window: this.store.graphFloatTensor(stepName, muWindow, [1, 100, this.manifest.positionContext + count]),
            cond_vec: this.store.graphFloatTensor(stepName, reference.condVec, [1, reference.condVec.length]),
            cond_mel_window: this.store.graphFloatTensor(stepName, condWindow, [1, 100, this.manifest.positionContext + count]),
            cond_mask_window: this.store.graphFloatTensor(stepName, maskWindow, [1, 1, this.manifest.positionContext + count]),
            cos: this.store.graphFloatTensor(stepName, r.cos, [count, this.config.model.acoustic_dit_dim_head]),
            sin: this.store.graphFloatTensor(stepName, r.sin, [count, this.config.model.acoustic_dit_dim_head]),
            x_context: old.x, past_k: old.k, past_v: old.v,
          });
          acousticState = { cached: end, x: out.x_context_out, k: out.present_k, v: out.present_v, shared: false };
          if (!old.shared) disposeAcousticState(old);
          normalized = Float32Array.from(await tensorData(out.mel)); dispose(out.mel);
        }
        const skipFrames = Math.max(0, pMel - start), audioFrames = count - skipFrames;
        if (audioFrames > 0) {
          const mel = channelSlice(normalized, 100, count, skipFrames, count);
          for (let m = 0; m < 100; m++) for (let t = 0; t < audioFrames; t++) mel[m * audioFrames + t] = mel[m * audioFrames + t] * std[m] + mean[m];
          const audio = await feedVocoder(mel, audioFrames); if (audio.length) { emitted += audio.length; chunks.push(audio); }
        }
        await cooperate(parameters, 0, 1);
      }
      return chunks;
    };
    const emit = (audio) => {
      const scaled = new Float32Array(audio.length);
      for (let i = 0; i < audio.length; i++) scaled[i] = audio[i] * gain;
      const out = gate(scaled);
      return out && out.length ? softLimit(out) : null;
    };
    for await (const semantic of this._semanticStream(text, reference, parameters, promptTokens)) {
      generated.push(...semantic);
      for (const audio of await advance(false)) { const out = emit(audio); if (out) yield out; }
    }
    for (const audio of await advance(true)) { const out = emit(audio); if (out) yield out; }
    if (vocoderState) {
      const flush = await this.store.session(vocoderName);
      const tailOut = await flush.run({
        is_start: vocoderControls.continue, is_flush: vocoderControls.flush,
        mel: this.store.graphFloatTensor(vocoderName, new Float32Array(0), [1, 100, 0]),
        embed_state: vocoderState.embed, conv_state: vocoderState.conv,
        pending0: vocoderState.pending0, pending1: vocoderState.pending1,
      }, ['istft_features']);
      const featureFrames = tailOut.istft_features.dims[1], features = Float32Array.from(await tensorData(tailOut.istft_features));
      dispose(tailOut.istft_features);
      let audio = streamIstft.push(features, featureFrames, true);
      if (skipSamples) { const drop = Math.min(skipSamples, audio.length); audio = audio.slice(drop); skipSamples -= drop; }
      const remaining = Math.max(0, generated.length * this.tokenSamples - emitted);
      audio = audio.slice(0, remaining);
      if (audio.length) {
        const scaled = new Float32Array(audio.length);
        for (let i = 0; i < audio.length; i++) scaled[i] = audio[i] * gain;
        let out = gate(scaled);
        if (out && out.length) {
          out = fadeEdges(trimTrail(out, this.sampleRate), this.sampleRate, false, true, last ? FINAL_FADE_SECONDS : JOIN_FADE_SECONDS);
          yield softLimit(out);
        }
      } else if (pending !== null && pending.length) {
        yield softLimit(fadeEdges(pending, this.sampleRate, false, last, FINAL_FADE_SECONDS));
      }
    }
    return Uint32Array.from(generated);
    } finally {
      if (vocoderState && !vocoderState.shared) for (const name of ['embed', 'conv', 'pending0', 'pending1']) dispose(vocoderState[name]);
      Object.values(vocoderControls).forEach(dispose);
      if (!acousticState.shared) disposeAcousticState(acousticState);
    }
  }

  async dispose() {
    const states = [...this.promptStates.values()]; this.promptStates.clear();
    for (const pending of states) {
      try { disposeAcousticState(await pending); } catch {}
    }
    const vocoderStates = [...this.vocoderPromptStates.values()]; this.vocoderPromptStates.clear();
    for (const pending of vocoderStates) {
      try { disposeVocoderPromptState(await pending); } catch {}
    }
    await this.store.releaseAll();
    this.store.cache.clearMemory();
  }
}

export function encodeWav(samples, sampleRate = 24000) {
  const bytes = new ArrayBuffer(44 + samples.length * 2), view = new DataView(bytes);
  const text = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE'); text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) view.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767))), true);
  return new Blob([bytes], { type: 'audio/wav' });
}
