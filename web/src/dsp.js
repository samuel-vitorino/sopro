const TWO_PI = 2 * Math.PI;

function nextPow2(value) { let n = 1; while (n < value) n <<= 1; return n; }

function fftPow2(real, imag, inverse = false) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (inverse ? TWO_PI : -TWO_PI) / len;
    const wlr = Math.cos(angle), wli = Math.sin(angle);
    for (let offset = 0; offset < n; offset += len) {
      let wr = 1, wi = 0;
      for (let j = 0; j < len / 2; j++) {
        const even = offset + j, odd = even + len / 2;
        const tr = real[odd] * wr - imag[odd] * wi;
        const ti = real[odd] * wi + imag[odd] * wr;
        real[odd] = real[even] - tr; imag[odd] = imag[even] - ti;
        real[even] += tr; imag[even] += ti;
        const nextWr = wr * wlr - wi * wli;
        wi = wr * wli + wi * wlr; wr = nextWr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
}

function fftAny(input) {
  const n = input.length;
  if ((n & (n - 1)) === 0) {
    const real = Float64Array.from(input), imag = new Float64Array(n);
    fftPow2(real, imag); return [real, imag];
  }
  const m = nextPow2(2 * n - 1);
  const ar = new Float64Array(m), ai = new Float64Array(m);
  const br = new Float64Array(m), bi = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    const angle = Math.PI * ((k * k) % (2 * n)) / n;
    const c = Math.cos(angle), s = Math.sin(angle);
    ar[k] = input[k] * c; ai[k] = -input[k] * s;
    br[k] = c; bi[k] = s;
    if (k) { br[m - k] = c; bi[m - k] = s; }
  }
  fftPow2(ar, ai); fftPow2(br, bi);
  for (let i = 0; i < m; i++) {
    const r = ar[i] * br[i] - ai[i] * bi[i];
    ai[i] = ar[i] * bi[i] + ai[i] * br[i]; ar[i] = r;
  }
  fftPow2(ar, ai, true);
  const real = new Float64Array(n), imag = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const angle = Math.PI * ((k * k) % (2 * n)) / n;
    const c = Math.cos(angle), s = Math.sin(angle);
    real[k] = ar[k] * c + ai[k] * s;
    imag[k] = ai[k] * c - ar[k] * s;
  }
  return [real, imag];
}

function reflectIndex(index, length) {
  if (length <= 1) return 0;
  while (index < 0 || index >= length) index = index < 0 ? -index : 2 * length - index - 2;
  return index;
}

function hzToMel(hz, scale) {
  if (scale === 'htk') return 2595 * Math.log10(1 + hz / 700);
  const minLogHz = 1000, minLogMel = 15, step = Math.log(6.4) / 27;
  return hz < minLogHz ? 3 * hz / 200 : minLogMel + Math.log(hz / minLogHz) / step;
}
function melToHz(mel, scale) {
  if (scale === 'htk') return 700 * (10 ** (mel / 2595) - 1);
  const minLogMel = 15, step = Math.log(6.4) / 27;
  return mel < minLogMel ? 200 * mel / 3 : 1000 * Math.exp(step * (mel - minLogMel));
}

function melBank(sampleRate, nFft, nMels, fMin, fMax, scale, norm) {
  const bins = nFft / 2 + 1;
  const lo = hzToMel(fMin, scale), hi = hzToMel(fMax, scale);
  const points = Array.from({ length: nMels + 2 }, (_, i) => melToHz(lo + (hi - lo) * i / (nMels + 1), scale));
  const bank = Array.from({ length: nMels }, () => new Float32Array(bins));
  for (let m = 0; m < nMels; m++) {
    const left = points[m], center = points[m + 1], right = points[m + 2];
    const area = norm === 'slaney' ? 2 / (right - left) : 1;
    for (let k = 0; k < bins; k++) {
      const hz = sampleRate * k / nFft;
      bank[m][k] = area * Math.max(0, Math.min((hz - left) / (center - left), (right - hz) / (right - center)));
    }
  }
  return bank;
}

function melSpectrogram(wav, { sampleRate, nFft, winLength = nFft, hopLength, nMels, fMin = 0, fMax = sampleRate / 2, power = 2, scale = 'htk', norm = null, frames = null }) {
  const pad = nFft >> 1;
  const count = frames ?? (Math.floor(wav.length / hopLength) + 1);
  const bins = nFft / 2 + 1;
  const window = new Float64Array(nFft);
  const start = Math.floor((nFft - winLength) / 2);
  for (let i = 0; i < winLength; i++) window[start + i] = 0.5 - 0.5 * Math.cos(TWO_PI * i / winLength);
  const bank = melBank(sampleRate, nFft, nMels, fMin, fMax, scale, norm);
  const output = new Float32Array(nMels * count);
  const frame = new Float64Array(nFft);
  for (let t = 0; t < count; t++) {
    const base = t * hopLength - pad;
    for (let i = 0; i < nFft; i++) frame[i] = wav[reflectIndex(base + i, wav.length)] * window[i];
    const [real, imag] = fftAny(frame);
    const spectrum = new Float64Array(bins);
    for (let k = 0; k < bins; k++) {
      const magnitude2 = real[k] * real[k] + imag[k] * imag[k];
      spectrum[k] = power === 1 ? Math.sqrt(magnitude2) : magnitude2;
    }
    for (let m = 0; m < nMels; m++) {
      let sum = 0;
      for (let k = 0; k < bins; k++) sum += spectrum[k] * bank[m][k];
      output[m * count + t] = sum;
    }
  }
  return { data: output, frames: count };
}

export function resample(wav, sourceRate, targetRate) {
  if (sourceRate === targetRate) return Float32Array.from(wav);
  const length = Math.max(1, Math.round(wav.length * targetRate / sourceRate));
  const out = new Float32Array(length), radius = 16, ratio = sourceRate / targetRate;
  for (let i = 0; i < length; i++) {
    const center = (i + 0.5) * ratio - 0.5, base = Math.floor(center);
    let sum = 0, weightSum = 0;
    for (let tap = -radius + 1; tap <= radius; tap++) {
      const source = base + tap;
      if (source < 0 || source >= wav.length) continue;
      const x = center - source;
      const sinc = Math.abs(x) < 1e-8 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      const window = 0.5 + 0.5 * Math.cos(Math.PI * x / radius);
      const weight = sinc * window;
      sum += wav[source] * weight; weightSum += weight;
    }
    out[i] = weightSum ? sum / weightSum : 0;
  }
  return out;
}

function frameRms(wav, sampleRate) {
  const win = Math.floor(sampleRate * 0.025), hop = Math.floor(sampleRate * 0.01);
  const values = [];
  for (let start = 0; start + win <= wav.length; start += hop) {
    let sum = 0; for (let i = start; i < start + win; i++) sum += wav[i] * wav[i];
    values.push(Math.max(1e-6, Math.sqrt(sum / win)));
  }
  return { values, win, hop };
}
function quantile(values, q) {
  const a = Float64Array.from(values).sort();
  if (!a.length) return 1e-6;
  const position = q * (a.length - 1), low = Math.floor(position), weight = position - low;
  return low + 1 < a.length ? a[low] * (1 - weight) + a[low + 1] * weight : a[low];
}
function median(values) { const a = Float64Array.from(values).sort(); return a.length ? a[Math.floor((a.length - 1) / 2)] : 1e-6; }
class RoomToneRandom {
  constructor() { this.state = 1; this.spare = null; }
  value() { let x = this.state; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.state = x >>> 0; return (this.state + 0.5) / 4294967296; }
  normal() { if (this.spare !== null) { const v = this.spare; this.spare = null; return v; } const r = Math.sqrt(-2 * Math.log(Math.max(1e-12, this.value()))), a = 2 * Math.PI * this.value(); this.spare = r * Math.sin(a); return r * Math.cos(a); }
}

const PROMPT_LEVEL_DB = -19.8, OUTPUT_LEVEL_DB = -23.0, REF_GAIN_LIMIT_DB = 30.0, MIN_ACTIVE_SECONDS = 0.4;
const SAMPLE_RATE = 24000, BAND_LIMIT_HZ = 10900;
const PAUSE_MIN_SECONDS = 0.10, PAUSE_KEEP_SECONDS = 0.15, CROP_FORWARD_SECONDS = 5.0, CROP_BACKWARD_SECONDS = 5.0, MIN_KEEP_FRACTION = 0.75, ROOM_TONE_SECONDS = 0.25, FADE_SECONDS = 0.02;
const ONSET_THRESHOLD_DB = -45.0, ONSET_OVER_FLOOR_DB = 15.0, ONSET_WINDOW_FRAMES = 6, ONSET_MIN_FRAMES = 5;
export const LEAD_IN_SECONDS = 0.08, SEGMENT_LEAD_SECONDS = 0.30, SEGMENT_SKIP_SECONDS = 0.10, TRAIL_SECONDS = 0.30, JOIN_FADE_SECONDS = 0.01, FINAL_FADE_SECONDS = 0.08, GATE_HOLD_SECONDS = 2;

function pauseRuns(quiet, minRun) {
  const runs = [];
  for (let i = 0; i < quiet.length;) {
    if (!quiet[i]) { i++; continue; }
    let j = i; while (j < quiet.length && quiet[j]) j++;
    if (j - i >= minRun) runs.push([i, j]);
    i = j;
  }
  return runs;
}

function finishWithRoomTone(wav, sampleRate, floor) {
  const fade = Math.floor(FADE_SECONDS * sampleRate), out = new Float32Array(wav.length + Math.floor(ROOM_TONE_SECONDS * sampleRate));
  out.set(wav);
  for (let i = 0; i < fade; i++) out[wav.length - fade + i] *= 1 - i / (fade - 1);
  const random = new RoomToneRandom();
  for (let i = wav.length; i < out.length; i++) out[i] = random.normal() * floor;
  return out;
}

export function cropOnPause(wav, seconds, sampleRate) {
  const win = Math.floor(sampleRate * 0.025), hop = Math.floor(sampleRate * 0.010), total = wav.length;
  if (total < 4 * win) return wav;
  const { values } = frameRms(wav, sampleRate);
  const floor = quantile(values, 0.1);
  const runs = pauseRuns(values.map((value) => value < floor * 4), Math.max(1, Math.round(PAUSE_MIN_SECONDS / 0.010)));
  const keep = Math.round(PAUSE_KEEP_SECONDS / 0.010), target = Math.round(seconds * sampleRate);
  const cutAt = ([a, b]) => wav.slice(0, Math.min(total, (a + Math.min(b - a, keep)) * hop + win));
  if (total <= target) {
    if (runs.length && runs.at(-1)[1] * hop + win >= total - hop) return wav;
    const inside = runs.filter(([a]) => a * hop >= total * MIN_KEEP_FRACTION);
    return inside.length ? cutAt(inside.at(-1)) : finishWithRoomTone(wav, sampleRate, floor);
  }
  const forward = runs.filter(([a]) => a * hop >= target && a * hop <= target + CROP_FORWARD_SECONDS * sampleRate);
  if (forward.length) return cutAt(forward[0]);
  const backward = runs.filter(([a, b]) => a * hop < target && b * hop >= target - CROP_BACKWARD_SECONDS * sampleRate);
  if (backward.length) return cutAt(backward.at(-1));
  return finishWithRoomTone(wav.slice(0, target), sampleRate, floor);
}

export function speechLevelDb(wav, sampleRate) {
  const win = Math.floor(sampleRate * 0.025), hop = Math.floor(sampleRate * 0.010);
  if (wav.length < win) {
    let sum = 0; for (let i = 0; i < wav.length; i++) sum += wav[i] * wav[i];
    return [20 * Math.log10(Math.max(Math.sqrt(sum / Math.max(1, wav.length)), 1e-6)), 0];
  }
  const { values } = frameRms(wav, sampleRate);
  const threshold = quantile(values, 0.2) * 1.5, active = values.filter((value) => value > threshold);
  const kept = active.length ? active : values;
  return [20 * Math.log10(median(kept)), kept.length * hop / sampleRate];
}

export function normalizeReference(wav, sampleRate) {
  const [levelDb] = speechLevelDb(wav, sampleRate);
  let gainDb = Math.min(Math.max(PROMPT_LEVEL_DB - levelDb, 0), REF_GAIN_LIMIT_DB);
  let peak = 0;
  for (let i = 0; i < wav.length; i++) { const a = Math.abs(wav[i]); if (a > peak) peak = a; }
  if (peak > 0) gainDb = Math.min(gainDb, Math.max(0, 20 * Math.log10(0.95 / peak)));
  const gain = 10 ** (gainDb / 20);
  const out = new Float32Array(wav.length);
  for (let i = 0; i < wav.length; i++) out[i] = wav[i] * gain;
  return { wav: out, levelDb: levelDb + gainDb };
}

export function cropAndNormalizeReference(input, sampleRate, seconds = 10) {
  const cropped = cropOnPause(Float32Array.from(input), seconds, sampleRate);
  return normalizeReference(cropped, sampleRate);
}

export function outputGain(promptLevelDb = PROMPT_LEVEL_DB) { return 10 ** ((OUTPUT_LEVEL_DB - promptLevelDb) / 20); }

export function matchGain(wav, sampleRate, targetDb = OUTPUT_LEVEL_DB) {
  const [levelDb, seconds] = speechLevelDb(wav, sampleRate);
  if (seconds < MIN_ACTIVE_SECONDS) return outputGain();
  return 10 ** ((targetDb - levelDb) / 20);
}

function shortRms(wav, sampleRate) {
  const win = Math.floor(sampleRate * 0.010), frames = Math.floor(wav.length / win), values = new Float64Array(frames);
  for (let t = 0; t < frames; t++) {
    let sum = 0; for (let i = t * win; i < (t + 1) * win; i++) sum += wav[i] * wav[i];
    values[t] = Math.sqrt(sum / win);
  }
  return { values, win };
}

function onsetThreshold(values) {
  let threshold = 10 ** (ONSET_THRESHOLD_DB / 20);
  if (values.length >= 30) threshold = Math.max(threshold, quantile(values, 0.1) * 10 ** (ONSET_OVER_FLOOR_DB / 20));
  return threshold;
}

export function speechOnset(wav, sampleRate) {
  const { values, win } = shortRms(wav, sampleRate);
  if (wav.length < win * ONSET_WINDOW_FRAMES) return null;
  const threshold = onsetThreshold(values);
  for (let t = 0; t + ONSET_WINDOW_FRAMES <= values.length; t++) {
    let hits = 0; for (let i = 0; i < ONSET_WINDOW_FRAMES; i++) hits += values[t + i] > threshold ? 1 : 0;
    if (hits >= ONSET_MIN_FRAMES) return t * win;
  }
  return null;
}

export function energyOnset(wav, sampleRate, overFloorDb = 10, minFrames = 3) {
  const { values, win } = shortRms(wav, sampleRate);
  if (values.length < minFrames) return null;
  const threshold = Math.max(quantile(values, 0.1) * 10 ** (overFloorDb / 20), 10 ** (ONSET_THRESHOLD_DB / 20));
  let run = 0;
  for (let t = 0; t < values.length; t++) {
    run = values[t] > threshold ? run + 1 : 0;
    if (run >= minFrames) return (t - minFrames + 1) * win;
  }
  return null;
}

export function energyFraction(wav, sampleRate, start, end, overFloorDb = 10) {
  const { values, win } = shortRms(wav, sampleRate);
  if (end <= start || !values.length) return 0;
  const threshold = Math.max(quantile(values, 0.1) * 10 ** (overFloorDb / 20), 10 ** (ONSET_THRESHOLD_DB / 20));
  const a = Math.floor(start / win), b = Math.max(a + 1, Math.floor(end / win));
  const seg = values.slice(a, b);
  if (!seg.length) return 0;
  let hits = 0;
  for (const v of seg) hits += v > threshold ? 1 : 0;
  return hits / seg.length;
}

export function trimLead(wav, sampleRate, lead = LEAD_IN_SECONDS, skip = 0) {
  const onset = speechOnset(wav, sampleRate);
  if (onset === null) return wav;
  let cut = Math.max(onset - Math.floor(lead * sampleRate), Math.floor(skip * sampleRate));
  cut = Math.min(cut, Math.max(0, onset - Math.floor(0.02 * sampleRate)));
  return wav.slice(cut);
}

export function trimTrail(wav, sampleRate, trail = TRAIL_SECONDS) {
  const { values, win } = shortRms(wav, sampleRate);
  if (wav.length < win) return wav;
  const threshold = onsetThreshold(values);
  let last = -1;
  for (let t = 0; t < values.length; t++) if (values[t] > threshold) last = t;
  if (last < 0) return wav;
  return wav.slice(0, Math.min(wav.length, (last + 1) * win + Math.floor(trail * sampleRate)));
}

export function fadeEdges(wav, sampleRate, fadeIn, fadeOut, fadeSeconds = JOIN_FADE_SECONDS) {
  const fade = Math.floor(fadeSeconds * sampleRate);
  if (wav.length <= 2 * fade) return wav;
  const out = Float32Array.from(wav);
  for (let i = 0; i < fade; i++) {
    const ramp = i / (fade - 1);
    if (fadeIn) out[i] *= ramp;
    if (fadeOut) out[out.length - 1 - i] *= ramp;
  }
  return out;
}

export function joinSegments(parts, sampleRate) {
  const faded = parts.map((part, i) => fadeEdges(part, sampleRate, i > 0, i + 1 < parts.length));
  const out = new Float32Array(faded.reduce((sum, part) => sum + part.length, 0));
  let offset = 0; for (const part of faded) { out.set(part, offset); offset += part.length; }
  return out;
}

export function referenceFeatures(wav24) {
  const wav16 = resample(wav24, 24000, 16000);
  const semanticFrames = Math.ceil(wav16.length / 160);
  const padded = new Float32Array(wav16.length + 400); padded.set(wav16);
  const semantic = melSpectrogram(padded, { sampleRate: 16000, nFft: 400, hopLength: 160, nMels: 80, power: 2, scale: 'slaney', norm: 'slaney', frames: semanticFrames + 2 });
  let maxLog = -Infinity;
  for (let i = 0; i < semantic.data.length; i++) { semantic.data[i] = Math.log10(Math.max(1e-10, semantic.data[i])); maxLog = Math.max(maxLog, semantic.data[i]); }
  for (let i = 0; i < semantic.data.length; i++) semantic.data[i] = (Math.max(semantic.data[i], maxLog - 8) + 4) / 4;
  const nTokens = Math.ceil(wav24.length / 1024), n50 = Math.ceil(semanticFrames / 2);
  const left = new BigInt64Array(nTokens), right = new BigInt64Array(nTokens), weight = new Float32Array(nTokens);
  for (let i = 0; i < nTokens; i++) {
    const source = Math.max(0, Math.min(n50 - 1, (i + 0.5) * n50 / nTokens - 0.5));
    left[i] = BigInt(Math.floor(source)); right[i] = BigInt(Math.min(n50 - 1, Math.floor(source) + 1)); weight[i] = source - Math.floor(source);
  }
  const speaker = melSpectrogram(wav16, { sampleRate: 16000, nFft: 1024, winLength: 400, hopLength: 160, nMels: 80, fMin: 20, fMax: 7600, power: 2, scale: 'slaney', norm: 'slaney' });
  for (let t = 0; t < speaker.frames; t++) {
    let mean = 0;
    for (let m = 0; m < 80; m++) { const index = m * speaker.frames + t; speaker.data[index] = Math.log(Math.max(1e-5, speaker.data[index])); mean += speaker.data[index]; }
    mean /= 80; let variance = 0;
    for (let m = 0; m < 80; m++) variance += (speaker.data[m * speaker.frames + t] - mean) ** 2;
    const inv = 1 / Math.sqrt(variance / 80 + 1e-5);
    for (let m = 0; m < 80; m++) speaker.data[m * speaker.frames + t] = (speaker.data[m * speaker.frames + t] - mean) * inv;
  }
  const acoustic = melSpectrogram(wav24, { sampleRate: 24000, nFft: 1024, hopLength: 256, nMels: 100, power: 1, scale: 'htk' });
  for (let i = 0; i < acoustic.data.length; i++) acoustic.data[i] = Math.log(Math.max(1e-7, acoustic.data[i]));
  return { semantic, speaker, acoustic, interp: { left, right, weight }, nTokens };
}

function inverseSpectrum(features, offset, nFft = 1024) {
  const bins = nFft / 2 + 1, real = new Float64Array(nFft), imag = new Float64Array(nFft);
  const limit = Math.min(bins, Math.ceil(BAND_LIMIT_HZ * nFft / SAMPLE_RATE));
  for (let k = 0; k < limit; k++) {
    const mag = Math.exp(Math.min(Math.log(100), features[offset + k]));
    const phase = features[offset + bins + k];
    real[k] = mag * Math.cos(phase); imag[k] = mag * Math.sin(phase);
  }
  for (let k = 1; k < bins - 1; k++) { real[nFft - k] = real[k]; imag[nFft - k] = -imag[k]; }
  fftPow2(real, imag, true); return real;
}

export function istft(features, frames, nFft = 1024, hop = 256) {
  const size = Math.max(0, (frames - 1) * hop + nFft), ola = new Float64Array(size), env = new Float64Array(size);
  for (let t = 0; t < frames; t++) {
    const wave = inverseSpectrum(features, t * (nFft + 2), nFft), start = t * hop;
    for (let i = 0; i < nFft; i++) { const w = 0.5 - 0.5 * Math.cos(TWO_PI * i / nFft); ola[start + i] += wave[i] * w; env[start + i] += w * w; }
  }
  const pad = nFft / 2, out = new Float32Array(Math.max(0, size - 2 * pad));
  for (let i = 0; i < out.length; i++) out[i] = ola[i + pad] / Math.max(1e-8, env[i + pad]);
  return out;
}

export class StreamingISTFT {
  constructor(nFft = 1024, hop = 256) { this.nFft = nFft; this.hop = hop; this.pad = nFft / 2; this.processed = 0; this.emitted = 0; this.tailStart = 0; this.ola = new Float64Array(0); this.env = new Float64Array(0); }
  push(features, frames, flush = false) {
    const requiredEnd = frames ? (this.processed + frames - 1) * this.hop + this.nFft : this.tailStart + this.ola.length;
    const required = Math.max(this.ola.length, requiredEnd - this.tailStart);
    if (required > this.ola.length) { const a = new Float64Array(required), e = new Float64Array(required); a.set(this.ola); e.set(this.env); this.ola = a; this.env = e; }
    for (let t = 0; t < frames; t++) {
      const wave = inverseSpectrum(features, t * (this.nFft + 2), this.nFft);
      const start = (this.processed + t) * this.hop - this.tailStart;
      for (let i = 0; i < this.nFft; i++) { const w = 0.5 - 0.5 * Math.cos(TWO_PI * i / this.nFft); this.ola[start + i] += wave[i] * w; this.env[start + i] += w * w; }
    }
    this.processed += frames;
    let target = Math.max(0, this.processed * this.hop - this.pad);
    if (flush && this.processed) target = Math.max(target, (this.processed - 1) * this.hop);
    const count = Math.max(0, target - this.emitted), rawStart = this.emitted + this.pad - this.tailStart;
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = this.ola[rawStart + i] / Math.max(1e-8, this.env[rawStart + i]);
    this.emitted = target;
    const trim = this.emitted + this.pad - this.tailStart;
    this.ola = this.ola.slice(trim); this.env = this.env.slice(trim); this.tailStart += trim;
    return out;
  }
}

export function softLimit(samples, gain = 1) {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) { const x = samples[i] * gain, a = Math.abs(x); out[i] = a > 0.9 ? Math.sign(x) * (0.9 + 0.1 * Math.tanh((a - 0.9) / 0.1)) : x; }
  return out;
}
