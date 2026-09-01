import { SoproTTS } from '@soprotts/onnx-web';

const mobile = navigator.userAgentData?.mobile ?? (/Mobi|Android/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1));
const modelParameter = new URLSearchParams(location.search).get('model');
const modelBaseUrl = modelParameter ? new URL(modelParameter, location.href).href.replace(/\/$/, '') : null;

export const runtimeUi = {
  title: 'Sopro ONNX',
  description: 'Local ONNX voice synthesis',
  loaderTitle: 'Downloading ONNX model',
  loaderStatus: 'Preparing local runtime…',
  modelUrl: modelBaseUrl ?? 'https://huggingface.co/samuel-vitorino/sopro-v2-turbo-onnx',
  alwaysShowLoader: false,
  selector: !mobile && navigator.gpu ? {
    label: 'Backend',
    storageKey: 'sopro:onnx-backend-v1',
    options: [['webgpu', 'WebGPU'], ['wasm', 'WASM']],
  } : null,
};

export async function createRuntime(options = {}) {
  const runtime = await SoproTTS.create({
    modelBaseUrl,
    token: localStorage.getItem('sopro:hf-token'),
    wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/',
    backend: options.variant,
    onProgress: options.onProgress,
  });
  runtime.variant = runtime.backend.startsWith('webgpu') ? 'webgpu' : 'wasm';
  runtime.cacheKey = `${runtime.manifest.model}:${runtime.manifest.revision ?? 'unversioned'}:${runtime.backend}`;
  return runtime;
}
