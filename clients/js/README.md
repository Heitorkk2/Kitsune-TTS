# Kitsune-TTS JS

JavaScript runtime for Kitsune-TTS using ONNX.

Licensed under GPL-3.0-only, consistently with the main Kitsune-TTS project.

## Files

- `kitsune-tts.js` — Loads and runs the ONNX model.
- `phonemizer.js` — Text normalization + rule-based PT-BR phonemizer.

## Usage

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js"></script>
<script src="phonemizer.js"></script>
<script src="kitsune-tts.js"></script>

<script>
  const tts = new KitsuneTTS();
  await tts.load('./kitsune39M.onnx');
  const audio = await tts.synthesize('Olá mundo!', 0);
  tts.play(audio);
</script>
```

WASM multithreading requires a cross-origin-isolated page. By default the
client lets ONNX Runtime select the thread count and uses its proxy worker to
keep the UI responsive. A manual configuration is also available:

```js
await tts.load('./kitsune39M.onnx', {
  provider: 'wasm',
  numThreads: 4,
  proxy: true,
  onProgress: (loaded, total, percent) => console.log(percent),
});
```

For WebGPU, load the WebGPU runtime bundle and request the provider. The client
falls back to WASM if session initialization fails:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.js"></script>
```

```js
await tts.load('./kitsune39M.onnx', { provider: 'webgpu' });
```

For long text, use `buildPhonemizedSegments()` together with
`synthesizeFromSegments()` to bound peak inference memory and preserve pauses.

## Speakers

| ID | Name     |
|----|----------|
| 0  | Emilia   |
| 1  | Frieren  |
| 2  | Zero Two |
| 3  | Violet   |
| 4  | Hiro     |
