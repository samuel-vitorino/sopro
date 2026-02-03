const $ = (id) => document.getElementById(id);

let aborter = null;
let isGenerating = false;

let audioCtx = null;
let analyser = null;
let outputGain = null;

let player = null;

let resampler = null;

let playbackPaused = false;

let lastOutput = {
    sr: null,
    f32: null,
};

let currentRefFile = null;
let currentRefUrl = null;

let isCachingRef = false;
let cacheAborter = null;

let cachedRef = {
    id: null,
    seconds: null,
    sig: null,
};

function fileSig(f) {
    return f ? `${f.name}|${f.size}|${f.lastModified}` : null;
}

function setCaching(on) {
    isCachingRef = on;
    const btn = $("gen_btn");
    if (on) {
        btn.textContent = "Caching reference";
        btn.disabled = true;
        btn.classList.remove("danger");
        btn.classList.add("primary");
    } else {
        if (!isGenerating) {
            btn.textContent = "Generate";
            btn.disabled = false;
            btn.classList.remove("danger");
            btn.classList.add("primary");
        }
    }
    updatePlaybackUI();
}

let rec = {
    active: false,
    stream: null,
    ctx: null,
    source: null,
    proc: null,
    gain0: null,
    analyser: null,
    chunks: [],
    total: 0,
    t0: 0,
    raf: null,
};

function logStatus(msg) {
    $("status").textContent = msg;
}

function toMaybeNumber(v) {
    const s = String(v ?? "").trim();
    if (!s) return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
}

function clampNumber(n, lo, hi, fallback) {
    n = Number(n);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(lo, Math.min(hi, n));
}

function setGenerating(on) {
    isGenerating = on;
    const btn = $("gen_btn");
    if (on) {
        btn.textContent = "Stop generating";
        btn.classList.remove("primary");
        btn.classList.add("danger");
    } else {
        btn.textContent = "Generate";
        btn.classList.remove("danger");
        btn.classList.add("primary");
    }
    updatePlaybackUI();
}

function updatePlaybackUI() {
    const playBtn = $("play_btn");
    const restartBtn = $("restart_btn");
    const downloadBtn = $("download_btn");

    const hasOutput = !!(lastOutput.f32 && lastOutput.sr);
    const canRestart = hasOutput && !isGenerating;
    const canDownload = hasOutput && !isGenerating;

    const canPlayToggle = (hasOutput || isGenerating) && !!audioCtx;

    if (restartBtn) restartBtn.disabled = !canRestart;
    if (downloadBtn) downloadBtn.disabled = !canDownload;

    if (playBtn) {
        playBtn.disabled = !canPlayToggle;
        playBtn.dataset.state = playbackPaused ? "paused" : "playing";
        setBtnText(playBtn, playbackPaused ? "Resume" : "Pause");
    }
}

function setBtnText(el, txt) {
    if (!el) return;
    const span = el.querySelector(".txt");
    if (span) span.textContent = txt;
    else el.textContent = txt;
}

function makeWorkletCode() {
    return `
class PCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.readIndex = 0;

    this.port.onmessage = (e) => {
      const { type, data } = e.data || {};
      if (type === "push" && data) {
          const arr = (data instanceof Float32Array) ? data : new Float32Array(data);
          this.queue.push(arr);
      } else if (type === "reset") {
        this.queue = [];
        this.current = null;
        this.readIndex = 0;
      }
    };
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const ch0 = out[0];
    let i = 0;

    while (i < ch0.length) {
      if (!this.current) {
        this.current = this.queue.shift() || null;
        this.readIndex = 0;
        if (!this.current) {
          for (; i < ch0.length; i++) ch0[i] = 0;
          return true;
        }
      }

      const cur = this.current;
      const remain = cur.length - this.readIndex;
      const need = ch0.length - i;
      const take = Math.min(remain, need);

      ch0.set(cur.subarray(this.readIndex, this.readIndex + take), i);
      i += take;
      this.readIndex += take;

      if (this.readIndex >= cur.length) {
        this.current = null;
      }
    }
    return true;
  }
}
registerProcessor("pcm-player", PCMPlayer);
`;
}

function concatFloat(a, b) {
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
}

function createResampler(inSR, outSR) {
    inSR = Number(inSR);
    outSR = Number(outSR);
    if (!Number.isFinite(inSR) || !Number.isFinite(outSR) || inSR <= 0 || outSR <= 0) {
        return { process: (x) => x, reset: () => { } };
    }
    if (inSR === outSR) {
        return { process: (x) => x, reset: () => { } };
    }

    const step = inSR / outSR;
    let pos = 0;
    let carry = new Float32Array(0);

    return {
        process(chunk) {
            if (!chunk || chunk.length === 0) return new Float32Array(0);
            const input = carry.length ? concatFloat(carry, chunk) : chunk;

            const maxOut = Math.floor((input.length - pos - 1) / step);
            if (maxOut <= 0) {
                const keepFrom = Math.max(0, Math.floor(pos) - 1);
                carry = input.slice(keepFrom);
                pos = pos - keepFrom;
                return new Float32Array(0);
            }

            const out = new Float32Array(maxOut);
            for (let i = 0; i < maxOut; i++) {
                const p = pos + i * step;
                const i0 = Math.floor(p);
                const frac = p - i0;
                const s0 = input[i0];
                const s1 = input[i0 + 1];
                out[i] = s0 * (1 - frac) + s1 * frac;
            }

            pos = pos + maxOut * step;

            const keepFrom = Math.max(0, Math.floor(pos) - 1);
            carry = input.slice(keepFrom);
            pos = pos - keepFrom;

            return out;
        },
        reset() { pos = 0; carry = new Float32Array(0); },
    };
}

function pcm16ToFloat32(pcmBytes) {
    const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);
    const n = pcmBytes.byteLength / 2;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true) / 32768;
    return out;
}

function parseWavPcm16Mono(arrayBuffer) {
    const dv = new DataView(arrayBuffer);
    const u8 = new Uint8Array(arrayBuffer);

    function str(off, len) {
        let s = "";
        for (let i = 0; i < len; i++) s += String.fromCharCode(u8[off + i]);
        return s;
    }

    if (str(0, 4) !== "RIFF" || str(8, 4) !== "WAVE") {
        throw new Error("Invalid WAV (missing RIFF/WAVE).");
    }

    let fmt = null;
    let dataOff = null;
    let dataLen = null;

    let off = 12;
    while (off + 8 <= dv.byteLength) {
        const id = str(off, 4);
        const size = dv.getUint32(off + 4, true);
        const body = off + 8;

        if (id === "fmt ") {
            const audioFormat = dv.getUint16(body + 0, true);
            const numChannels = dv.getUint16(body + 2, true);
            const sampleRate = dv.getUint32(body + 4, true);
            const bitsPerSample = dv.getUint16(body + 14, true);
            fmt = { audioFormat, numChannels, sampleRate, bitsPerSample };
        } else if (id === "data") {
            dataOff = body;
            dataLen = size;
            break;
        }

        off = body + size + (size % 2);
    }

    if (!fmt || dataOff == null || dataLen == null) {
        throw new Error("Invalid WAV (missing fmt/data).");
    }
    if (fmt.audioFormat !== 1) throw new Error("WAV must be PCM (format=1).");
    if (fmt.numChannels !== 1) throw new Error("WAV must be mono.");
    if (fmt.bitsPerSample !== 16) throw new Error("WAV must be 16-bit PCM.");

    const pcm = new Uint8Array(arrayBuffer, dataOff, dataLen);
    return { sr: fmt.sampleRate, pcmBytes: pcm };
}

async function cacheReferenceNow() {
    const f = currentRefFile;
    if (!f) return;

    if (cacheAborter) cacheAborter.abort();
    cacheAborter = new AbortController();

    cachedRef.id = null;
    cachedRef.seconds = null;
    cachedRef.sig = fileSig(f);

    setCaching(true);
    logStatus("Caching reference on server...");

    try {
        const fd = new FormData();
        fd.append("ref_audio", f);

        const rs = toMaybeNumber($("ref_seconds").value);
        if (rs !== null) fd.append("ref_seconds", String(rs));

        const resp = await fetch("/v1/reference/cache", {
            method: "POST",
            body: fd,
            signal: cacheAborter.signal,
        });

        if (!resp.ok) {
            const txt = await resp.text().catch(() => "");
            throw new Error(`HTTP ${resp.status}: ${txt || resp.statusText}`);
        }

        const js = await resp.json();
        cachedRef.id = js.ref_id;
        cachedRef.seconds = js.ref_seconds;

        logStatus(`Reference cached (ref_id=${cachedRef.id.slice(0, 10)}..., ref_seconds=${cachedRef.seconds})`);
    } catch (err) {
        if (String(err?.name) === "AbortError") return;
        logStatus(`Reference cache failed: ${err?.message || err}`);
    } finally {
        cacheAborter = null;
        setCaching(false);
    }
}

async function ensureAudio() {
    if (audioCtx && analyser && outputGain && player) return;

    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error("Web Audio API not available in this browser.");

    audioCtx = new AC({ latencyHint: "interactive" });

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.85;

    outputGain = audioCtx.createGain();
    outputGain.gain.value = 1.0;

    analyser.connect(outputGain);
    outputGain.connect(audioCtx.destination);

    if (audioCtx.audioWorklet && typeof audioCtx.audioWorklet.addModule === "function") {
        const code = makeWorkletCode();
        const blobUrl = URL.createObjectURL(new Blob([code], { type: "application/javascript" }));
        await audioCtx.audioWorklet.addModule(blobUrl);
        URL.revokeObjectURL(blobUrl);

        const node = new AudioWorkletNode(audioCtx, "pcm-player", {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [1],
        });

        node.connect(analyser);

        player = {
            push(f32) {
                if (!f32 || f32.length === 0) return;
                const copy = new Float32Array(f32);
                node.port.postMessage({ type: "push", data: copy.buffer }, [copy.buffer]);
            },
            reset() { node.port.postMessage({ type: "reset" }); },
        };

        startWaveLoop();
        updatePlaybackUI();
        return;
    }

    const queue = [];
    let cur = null;
    let idx = 0;

    const sp = audioCtx.createScriptProcessor(2048, 0, 1);
    sp.onaudioprocess = (e) => {
        const out = e.outputBuffer.getChannelData(0);
        let i = 0;
        while (i < out.length) {
            if (!cur) {
                cur = queue.shift() || null;
                idx = 0;
                if (!cur) {
                    for (; i < out.length; i++) out[i] = 0;
                    return;
                }
            }
            const take = Math.min(cur.length - idx, out.length - i);
            out.set(cur.subarray(idx, idx + take), i);
            i += take;
            idx += take;
            if (idx >= cur.length) cur = null;
        }
    };

    sp.connect(analyser);

    player = {
        push(f32) { if (f32 && f32.length) queue.push(f32); },
        reset() { queue.length = 0; cur = null; idx = 0; },
    };

    startWaveLoop();
    updatePlaybackUI();
}

let waveRAF = null;

function startWaveLoop() {
    if (waveRAF) return;
    const canvas = $("scope");
    const ctx2d = canvas.getContext("2d", { alpha: false });

    function resize() {
        const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
        ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    const data = new Uint8Array(2048);

    function draw() {
        waveRAF = requestAnimationFrame(draw);

        const w = canvas.getBoundingClientRect().width;
        const h = canvas.getBoundingClientRect().height;

        ctx2d.fillStyle = "rgba(0,0,0,0.92)";
        ctx2d.fillRect(0, 0, w, h);

        const a = analyser;

        if (!a) {
            ctx2d.strokeStyle = "rgba(255,255,255,0.28)";
            ctx2d.lineWidth = 2;
            ctx2d.beginPath();
            ctx2d.moveTo(0, h * 0.5);
            ctx2d.lineTo(w, h * 0.5);
            ctx2d.stroke();
            return;
        }

        a.getByteTimeDomainData(data);

        ctx2d.lineWidth = 3;
        ctx2d.strokeStyle = "rgba(255,255,255,0.10)";
        ctx2d.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = (i / (data.length - 1)) * w;
            const v = data[i] / 255;
            const y = (1 - v) * h;
            if (i === 0) ctx2d.moveTo(x, y);
            else ctx2d.lineTo(x, y);
        }
        ctx2d.stroke();

        ctx2d.lineWidth = 1.5;
        ctx2d.strokeStyle = "rgba(255,255,255,0.82)";
        ctx2d.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = (i / (data.length - 1)) * w;
            const v = data[i] / 255;
            const y = (1 - v) * h;
            if (i === 0) ctx2d.moveTo(x, y);
            else ctx2d.lineTo(x, y);
        }
        ctx2d.stroke();

        ctx2d.strokeStyle = "rgba(255,255,255,0.06)";
        ctx2d.lineWidth = 1;
        ctx2d.beginPath();
        ctx2d.moveTo(0, h * 0.5);
        ctx2d.lineTo(w, h * 0.5);
        ctx2d.stroke();
    }

    draw();
}

function setRefFile(file, label, previewUrl) {
    currentRefFile = file;

    $("ref_label").textContent =
        label || (file ? `${file.name} (${Math.round(file.size / 1024)} KB)` : "No reference selected yet.");

    const prev = $("ref_preview");
    if (currentRefUrl) {
        try { URL.revokeObjectURL(currentRefUrl); } catch { }
        currentRefUrl = null;
    }
    if (previewUrl) {
        currentRefUrl = previewUrl;
        prev.src = previewUrl;
        prev.style.display = "block";
    } else {
        prev.removeAttribute("src");
        prev.style.display = "none";
    }
}

$("ref_file").addEventListener("change", async () => {
    const f = $("ref_file").files?.[0] || null;
    if (!f) return;
    const url = URL.createObjectURL(f);
    setRefFile(f, `Reference: upload • ${f.name}`, url);
    await cacheReferenceNow();
});

function encodeWavPCM16(float32, sampleRate) {
    const numSamples = float32.length;
    const dataSize = numSamples * 2;

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeStr(off, s) { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); }

    writeStr(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, dataSize, true);

    let o = 44;
    for (let i = 0; i < numSamples; i++) {
        let s = Math.max(-1, Math.min(1, float32[i]));
        s = s < 0 ? s * 32768 : s * 32767;
        view.setInt16(o, s, true);
        o += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
}

async function startRecording() {
    if (rec.active) return;

    if (!navigator.mediaDevices?.getUserMedia) {
        logStatus("Error: getUserMedia() not available in this browser.");
        return;
    }

    const micBtn = $("mic_btn");
    micBtn.textContent = "Stop recording";
    micBtn.classList.add("danger");

    $("rec_timer").textContent = "00:00";

    try {
        rec.stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });

        const AC = window.AudioContext || window.webkitAudioContext;
        rec.ctx = new AC({ latencyHint: "interactive" });

        rec.analyser = rec.ctx.createAnalyser();
        rec.analyser.fftSize = 2048;
        rec.analyser.smoothingTimeConstant = 0.85;

        rec.source = rec.ctx.createMediaStreamSource(rec.stream);

        rec.proc = rec.ctx.createScriptProcessor(4096, 1, 1);
        rec.gain0 = rec.ctx.createGain();
        rec.gain0.gain.value = 0.0;

        rec.chunks = [];
        rec.total = 0;

        rec.proc.onaudioprocess = (e) => {
            const in0 = e.inputBuffer.getChannelData(0);
            const copy = new Float32Array(in0.length);
            copy.set(in0);
            rec.chunks.push(copy);
            rec.total += copy.length;
        };

        rec.source.connect(rec.analyser);
        rec.source.connect(rec.proc);
        rec.proc.connect(rec.gain0);
        rec.gain0.connect(rec.ctx.destination);

        rec.t0 = performance.now();
        rec.active = true;

        const tick = () => {
            if (!rec.active) return;
            const s = Math.floor((performance.now() - rec.t0) / 1000);
            const mm = String(Math.floor(s / 60)).padStart(2, "0");
            const ss = String(s % 60).padStart(2, "0");
            $("rec_timer").textContent = `${mm}:${ss}`;
            rec.raf = requestAnimationFrame(tick);
        };
        tick();

        logStatus("Recording mic reference…");
    } catch (err) {
        rec.active = false;
        micBtn.textContent = "Record mic";
        micBtn.classList.remove("danger");
        logStatus(`Error starting mic: ${err?.message || err}`);
    }
}

async function stopRecording() {
    if (!rec.active) return;

    rec.active = false;
    if (rec.raf) cancelAnimationFrame(rec.raf);

    const micBtn = $("mic_btn");
    micBtn.textContent = "Record mic";
    micBtn.classList.remove("danger");

    try {
        const merged = new Float32Array(rec.total);
        let off = 0;
        for (const c of rec.chunks) { merged.set(c, off); off += c.length; }

        const sr = rec.ctx.sampleRate || 48000;
        const wavBlob = encodeWavPCM16(merged, sr);
        const file = new File([wavBlob], "ref_mic.wav", { type: "audio/wav" });

        $("ref_file").value = "";
        const url = URL.createObjectURL(file);

        setRefFile(file, `Reference: mic • ${$("rec_timer").textContent}`, url);
        await cacheReferenceNow();
        logStatus("Mic reference cached. Click Generate.");
    } catch (err) {
        logStatus(`Error finalizing recording: ${err?.message || err}`);
    } finally {
        try { rec.proc?.disconnect(); } catch { }
        try { rec.source?.disconnect(); } catch { }
        try { rec.gain0?.disconnect(); } catch { }
        try { rec.stream?.getTracks()?.forEach(t => t.stop()); } catch { }
        try { await rec.ctx?.close(); } catch { }
        rec.stream = rec.ctx = rec.source = rec.proc = rec.gain0 = rec.analyser = null;
        rec.chunks = [];
        rec.total = 0;
        $("rec_timer").textContent = "";
    }
}

$("mic_btn").addEventListener("click", async () => {
    if (rec.active) await stopRecording();
    else await startRecording();
});

$("play_btn").addEventListener("click", async () => {
    await ensureAudio();

    try {
        if (!playbackPaused) {
            await audioCtx.suspend();
            playbackPaused = true;
            logStatus("Playback stopped.");
        } else {
            await audioCtx.resume();
            playbackPaused = false;
            logStatus("Playback resumed.");
        }
    } catch (e) {
        logStatus(`Playback toggle error: ${e?.message || e}`);
    } finally {
        updatePlaybackUI();
    }
});

$("ref_seconds").addEventListener("change", async () => {
    if (!currentRefFile) return;
    await cacheReferenceNow();
});

$("restart_btn").addEventListener("click", async () => {
    if (!lastOutput.f32 || !lastOutput.sr) return;
    if (isGenerating) return;

    await ensureAudio();
    await audioCtx.resume().catch(() => { });
    playbackPaused = false;

    if (player) player.reset();
    resampler = createResampler(lastOutput.sr, audioCtx.sampleRate);

    const chunkSize = 8192;
    for (let i = 0; i < lastOutput.f32.length; i += chunkSize) {
        let chunk = lastOutput.f32.subarray(i, i + chunkSize);
        chunk = resampler.process(chunk);
        player.push(chunk);
    }

    logStatus("Restarted from beginning.");
    updatePlaybackUI();
});

function stopGeneratingNow() {
    if (aborter) aborter.abort();
    aborter = null;
    if (player) player.reset();
    resampler = null;
}

$("gen_btn").addEventListener("click", async () => {
    if (isGenerating) {
        stopGeneratingNow();
        setGenerating(false);
        logStatus("Stopped generating.");
        return;
    }

    if (rec.active) await stopRecording();

    const text = $("text").value || "";
    if (!text.trim()) {
        logStatus("Please enter some text.");
        return;
    }

    const fileFromInput = $("ref_file").files?.[0] || null;
    const refFile = currentRefFile || fileFromInput;
    if (!refFile) {
        logStatus("Please upload a reference audio or record one from the microphone.");
        return;
    }

    setGenerating(true);

    await ensureAudio();
    if (!playbackPaused) {
        await audioCtx.resume().catch(() => { });
    }

    const stream = $("stream").checked;

    if (isCachingRef) {
        logStatus("Still caching reference…");
        return;
    }

    const fd = new FormData();
    fd.append("input", text);
    fd.append("stream", String(stream));

    const wantSig = fileSig(refFile);
    const wantRs = toMaybeNumber($("ref_seconds").value);
    const effWantRs = (wantRs === null) ? 12.0 : wantRs;

    const canUseCache =
        cachedRef.id &&
        cachedRef.sig === wantSig &&
        cachedRef.seconds != null &&
        Math.abs(Number(cachedRef.seconds) - Number(effWantRs)) < 1e-6;

    if (canUseCache) {
        fd.append("ref_id", cachedRef.id);
        fd.append("ref_seconds", String(cachedRef.seconds));
    } else {
        fd.append("ref_audio", refFile);
        if (wantRs !== null) fd.append("ref_seconds", String(wantRs));
    }

    fd.append("top_p", String(clampNumber($("top_p").value, 0.01, 1.0, 0.9)));
    fd.append("temperature", String(clampNumber($("temperature").value, 0.05, 3.0, 1.05)));
    fd.append("max_frames", String(clampNumber($("max_frames").value, 1, 2000, 400)));

    fd.append("style_strength", String(clampNumber($("style_strength").value, 0.0, 3.0, 1.0)));

    fd.append("anti_loop", String(!!$("anti_loop").checked));

    aborter = new AbortController();
    const t0 = performance.now();
    logStatus(stream ? "Requesting stream..." : "Requesting WAV...");

    lastOutput.sr = null;
    lastOutput.f32 = null;
    updatePlaybackUI();

    try {
        const resp = await fetch("/v1/audio/speech", {
            method: "POST",
            body: fd,
            signal: aborter.signal,
        });

        if (!resp.ok) {
            const txt = await resp.text().catch(() => "");
            throw new Error(`HTTP ${resp.status}: ${txt || resp.statusText}`);
        }

        if (!stream) {
            const buf = await resp.arrayBuffer();
            const { sr, pcmBytes } = parseWavPcm16Mono(buf);
            const f32 = pcm16ToFloat32(pcmBytes);

            lastOutput.sr = sr;
            lastOutput.f32 = f32;
            updatePlaybackUI();

            if (player) player.reset();
            resampler = createResampler(sr, audioCtx.sampleRate);

            const chunkSize = 8192;
            for (let i = 0; i < f32.length; i += chunkSize) {
                let chunk = f32.subarray(i, i + chunkSize);
                chunk = resampler.process(chunk);
                player.push(chunk);
            }

            logStatus(`WAV received. Total ${(performance.now() - t0).toFixed(0)} ms`);
            return;
        }

        if (player) player.reset();

        const reader = resp.body.getReader();
        let buffer = new Uint8Array(0);

        let headerRead = false;
        let remoteSR = 24000;
        let firstAudioAt = null;

        const streamChunks = [];
        let streamTotal = 0;

        function concatBytes(a, b) {
            const out = new Uint8Array(a.length + b.length);
            out.set(a, 0);
            out.set(b, a.length);
            return out;
        }

        function tryReadHeader() {
            if (buffer.length < 12) return false;
            const magic = String.fromCharCode(buffer[0], buffer[1], buffer[2], buffer[3]);
            if (magic !== "SPRO") throw new Error("Bad stream header (magic mismatch).");
            const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
            remoteSR = dv.getUint32(4, true);
            buffer = buffer.slice(12);

            resampler = createResampler(remoteSR, audioCtx.sampleRate);
            headerRead = true;
            return true;
        }

        function tryReadFrame() {
            if (buffer.length < 4) return null;
            const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
            const n = dv.getUint32(0, true);
            if (buffer.length < 4 + n) return null;
            const payload = buffer.slice(4, 4 + n);
            buffer = buffer.slice(4 + n);
            return payload;
        }

        logStatus("Streaming... waiting for first audio chunk");

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (value) buffer = concatBytes(buffer, value);

            if (!headerRead) {
                if (!tryReadHeader()) continue;
            }

            while (true) {
                const payload = tryReadFrame();
                if (!payload) break;

                if (firstAudioAt === null) {
                    firstAudioAt = performance.now();
                    const ttfa = (firstAudioAt - t0).toFixed(0);
                    logStatus(`Streaming... TTFA ${ttfa} ms (sr=${remoteSR}, ctxSR=${audioCtx.sampleRate})`);
                }

                const f32Remote = pcm16ToFloat32(payload);
                if (f32Remote.length) {
                    streamChunks.push(f32Remote);
                    streamTotal += f32Remote.length;
                }

                let f32Play = f32Remote;
                if (resampler) f32Play = resampler.process(f32Play);
                player.push(f32Play);
            }
        }

        if (streamTotal > 0) {
            const merged = new Float32Array(streamTotal);
            let off = 0;
            for (const c of streamChunks) { merged.set(c, off); off += c.length; }
            lastOutput.sr = remoteSR;
            lastOutput.f32 = merged;
            updatePlaybackUI();
        }

        logStatus(`Stream ended. Total ${(performance.now() - t0).toFixed(0)} ms`);
    } catch (err) {
        if (String(err?.name) === "AbortError") {
            logStatus("Stopped generating.");
            return;
        }
        logStatus(`Error: ${err?.message || err}`);
    } finally {
        aborter = null;
        setGenerating(false);
        updatePlaybackUI();
    }
});

$("download_btn").addEventListener("click", async () => {
    if (!lastOutput.f32 || !lastOutput.sr) return;

    try {
        const wavBlob = encodeWavPCM16(lastOutput.f32, lastOutput.sr);
        const url = URL.createObjectURL(wavBlob);

        const a = document.createElement("a");
        a.href = url;
        a.download = `soprotts_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
        document.body.appendChild(a);
        a.click();
        a.remove();

        setTimeout(() => {
            try { URL.revokeObjectURL(url); } catch { }
        }, 5000);
    } catch (e) {
        logStatus(`Download error: ${e?.message || e}`);
    }
});

startWaveLoop();
updatePlaybackUI();
