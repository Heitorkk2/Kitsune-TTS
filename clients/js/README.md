# Kitsune-TTS JS

JavaScript runtime for Kitsune-TTS using ONNX.

The Python API's optional `fast_cpu=True` vocoder is PyTorch-only. It does not
change this client or the ONNX model; continue using the standard ONNX export.

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

### Progressive audio

`streamFromSegments()` yields each PCM chunk as soon as it is ready, including
the configured silence gaps. Unlike `synthesizeFromSegments()`, it does not
retain and concatenate the entire utterance. Existing synthesis methods are
unchanged. This is segment-level streaming, not streaming within a segment;
splitting text can change prosody.

Schedule chunks on the Web Audio timeline so they do not overlap. Run this from
a user interaction to satisfy browser audio permissions:

```js
const context = new AudioContext();
await context.resume();
let nextStart = context.currentTime;
for await (const pcm of tts.streamFromSegments(buildPhonemizedSegments(text), 1)) {
  const buffer = context.createBuffer(1, pcm.length, tts.sampleRate);
  buffer.copyToChannel(pcm, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.onended = () => source.disconnect();
  nextStart = Math.max(nextStart, context.currentTime + 0.02);
  source.start(nextStart);
  nextStart += buffer.duration;
}
// Close the context when playback is finished and the player is no longer needed.
```

Breaking out of the loop prevents synthesis of subsequent segments; already
scheduled audio must be stopped separately by the player. If synthesis cannot
keep up with playback, gaps are still possible. Consumer/playback waiting time
is not an inference benchmark; use `synthesizeFromSegments()` for its existing
generation-time and RTF measurements.

## Speakers

| ID | Name     |
|----|----------|
| 0  | Emilia   |
| 1  | Frieren  |
| 2  | Zero Two |
| 3  | Violet   |
| 4  | Hiro     |
