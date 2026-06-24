/**
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Kitsune-TTS — Client-side ONNX inference engine (Pure JavaScript).
 *
 * Runs the VITS2 model entirely in the browser via onnxruntime-web (WASM).
 * No server required — the model is downloaded and executed locally.
 *
 * Expected ONNX graph I/O (as exported by export_onnx.py):
 *   inputs:  x [1, seq_len] int64, x_lengths [1] int64, sid [1] int64,
 *            noise_scale [1] float32, length_scale [1] float32
 *   output:  audio [1, 1, audio_len] float32 @ 22050 Hz
 *
 * Usage (browser <script> tag):
 *   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.js"></script>
 *   <script src="phonemizer.js"></script>
 *   <script src="kitsune-tts.js"></script>
 *   <script>
 *     const tts = new KitsuneTTS();
 *     await tts.load('./kitsune39M.onnx');
 *     const audio = await tts.synthesize('Olá mundo!', 0);
 *     tts.play(audio);
 *   </script>
 *
 * Usage (Node.js / CommonJS):
 *   const { KitsuneTTS } = require('./kitsune-tts.js');
 */

const SAMPLE_RATE = 22050;

function resolveTextToSequence() {
    if (typeof textToSequence === 'function') return textToSequence;
    if (typeof module !== 'undefined' && module.exports && typeof require === 'function') {
        return require('./phonemizer.js').textToSequence;
    }
    throw new Error('phonemizer.js not loaded (textToSequence is unavailable).');
}

async function readResponseBytes(response, onProgress) {
    const contentLength = response.headers.get('content-length');
    const totalBytes = contentLength ? Number.parseInt(contentLength, 10) : 0;

    // The native path avoids retaining a chunk list when progress reporting is
    // not needed. This is the lowest-peak-memory option supported by fetch().
    if (!onProgress || !response.body?.getReader) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (onProgress) {
            const mb = bytes.byteLength / (1024 * 1024);
            onProgress(Number(mb.toFixed(1)), totalBytes ? Number((totalBytes / (1024 * 1024)).toFixed(1)) : 0, 100);
        }
        return bytes;
    }

    const reader = response.body.getReader();
    const chunks = totalBytes > 0 ? null : [];
    let modelBuffer = totalBytes > 0 ? new Uint8Array(totalBytes) : null;
    let loadedBytes = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;

        if (modelBuffer) {
            if (loadedBytes + value.length > modelBuffer.length) {
                // Be robust to servers whose Content-Length describes a
                // compressed transfer rather than the decoded response body.
                const grown = new Uint8Array(Math.max(loadedBytes + value.length, modelBuffer.length * 2));
                grown.set(modelBuffer.subarray(0, loadedBytes));
                modelBuffer = grown;
            }
            modelBuffer.set(value, loadedBytes);
        } else {
            chunks.push(value);
        }

        loadedBytes += value.length;
        const loadedMB = Number((loadedBytes / (1024 * 1024)).toFixed(1));
        const totalMB = totalBytes > 0 ? Number((totalBytes / (1024 * 1024)).toFixed(1)) : 0;
        const percent = totalBytes > 0 ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100)) : 0;
        onProgress(loadedMB, totalMB, percent);
    }

    if (modelBuffer) {
        return loadedBytes === modelBuffer.length ? modelBuffer : modelBuffer.slice(0, loadedBytes);
    }

    modelBuffer = new Uint8Array(loadedBytes);
    let offset = 0;
    for (const chunk of chunks) {
        modelBuffer.set(chunk, offset);
        offset += chunk.length;
    }
    return modelBuffer;
}

class KitsuneTTS {
    constructor() {
        /** @type {any} */ this.session = null;
        /** @type {any} */ this.ortApi = null;
        this.sampleRate = SAMPLE_RATE;
        this.activeProvider = 'wasm';
        this.audioContext = null;
    }

    /**
     * Downloads and initializes the ONNX model.
     *
     * @param {string} modelUrl  URL or path to the .onnx file.
     * @param {object} [opts]
     * @param {number} [opts.numThreads]  WASM threads (default: 0, chosen by ORT).
     * @param {string} [opts.provider]    Execution provider ('wasm' | 'webgpu').
     * @param {boolean} [opts.proxy]      Run WASM inference off the UI thread.
     * @param {boolean} [opts.warmup]     Run one short inference after loading.
     * @param {boolean} [opts.fallbackToWasm] Fall back if WebGPU initialization fails.
     * @param {AbortSignal} [opts.signal] Abort signal for the model download.
     * @param {(loadedMB: number, totalMB: number, percent: number) => void} [opts.onProgress]
     *        Called periodically during model download.
     */
    async load(modelUrl, {
        numThreads = null,
        provider = 'wasm',
        proxy = true,
        warmup = true,
        fallbackToWasm = true,
        signal = undefined,
        onProgress = null,
    } = {}) {
        // Resolve the ORT runtime
        if (typeof ort !== 'undefined') {
            this.ortApi = ort;
        } else if (typeof module !== 'undefined' && module.exports && typeof require === 'function') {
            try {
                this.ortApi = require('onnxruntime-web');
            } catch (error) {
                throw new Error(
                    'onnxruntime-web is not installed. Add it before calling load().',
                    { cause: error }
                );
            }
        } else {
            throw new Error(
                'onnxruntime-web (ort) not found. Include the <script> tag or import it before this file.'
            );
        }

        let threads = numThreads === null ? 0 : Math.max(0, Math.trunc(numThreads));
        const isBrowserMainThread = typeof window !== 'undefined' && typeof document !== 'undefined';
        if (typeof crossOriginIsolated !== 'undefined' && !crossOriginIsolated && threads > 1) {
            console.warn('WASM multithreading requires a cross-origin-isolated page; using one thread.');
            threads = 1;
        }

        if (this.ortApi.env && this.ortApi.env.wasm) {
            this.ortApi.env.wasm.numThreads = threads;
            this.ortApi.env.wasm.proxy = Boolean(proxy && isBrowserMainThread && provider === 'wasm');
        }
        if (this.ortApi.env) {
            this.ortApi.env.logLevel = 'error';
        }

        this.activeProvider = provider;

        // Download with progress tracking
        const response = await fetch(modelUrl, { signal });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status} ${response.statusText} loading model (${modelUrl})`);
        }

        const modelBuffer = await readResponseBytes(response, onProgress);

        console.log('Download complete. Optimizing graph and initializing ONNX session...');

        const createSession = executionProvider => this.ortApi.InferenceSession.create(modelBuffer, {
            executionProviders: [executionProvider],
            graphOptimizationLevel: 'all',
            intraOpNumThreads: threads,
            interOpNumThreads: 1,
        });

        try {
            this.session = await createSession(provider);
        } catch (error) {
            if (provider !== 'webgpu' || !fallbackToWasm) throw error;
            console.warn(`WebGPU initialization failed; falling back to WASM: ${error.message}`);
            this.activeProvider = 'wasm';
            if (this.ortApi.env?.wasm) {
                this.ortApi.env.wasm.proxy = Boolean(proxy && isBrowserMainThread);
            }
            this.session = await createSession('wasm');
        }

        // JIT Warmup: pre-compile the WASM kernels with a realistic-length dummy input
        if (warmup) try {
            console.log('Warming up model (JIT compile)...');
            // A short, valid ", a ," sequence avoids treating 50 padding IDs
            // as real phonemes and generating a needlessly large warmup audio.
            await this.synthesizeFromIds([3, 16, 17, 16, 3], 0);
            console.log('Warmup complete!');
        } catch (e) {
            console.warn('Harmless warmup error (can be ignored):', e);
        }

        console.log(
            `Kitsune-TTS loaded locally via ${this.activeProvider.toUpperCase()} | ` +
            `inputs: [${this.session.inputNames}] | outputs: [${this.session.outputNames}]`
        );
    }

    /**
     * Synthesizes audio from pre-computed phoneme IDs.
     * This is the low-level function — use synthesize() for text input.
     *
     * @param {number[]} ids        Phoneme ID sequence.
     * @param {number}   [speakerId=0]
     * @param {object}   [opts]
     * @param {number}   [opts.noiseScale=0.667]
     * @param {number}   [opts.lengthScale=1.0]
     * @returns {Promise<Float32Array>} Raw PCM audio samples @ 22050 Hz.
     */
    async synthesizeFromIds(ids, speakerId = 0, { noiseScale = 0.667, lengthScale = 1.0 } = {}) {
        if (!this.session) throw new Error('Model not loaded — call load() first.');
        if (ids.length === 0) return new Float32Array(0);

        const x = new this.ortApi.Tensor('int64', BigInt64Array.from(ids.map(id => BigInt(id))), [1, ids.length]);
        const xLengths = new this.ortApi.Tensor('int64', BigInt64Array.from([BigInt(ids.length)]), [1]);
        const sid = new this.ortApi.Tensor('int64', BigInt64Array.from([BigInt(speakerId)]), [1]);
        const noiseScaleT = new this.ortApi.Tensor('float32', Float32Array.from([noiseScale]), [1]);
        const lengthScaleT = new this.ortApi.Tensor('float32', Float32Array.from([lengthScale]), [1]);

        const feeds = {
            x, x_lengths: xLengths, sid,
            noise_scale: noiseScaleT, length_scale: lengthScaleT,
        };

        const results = await this.session.run(feeds);
        // Clone the array to release the WASM memory reference and prevent leaks
        return new Float32Array(results.audio.data);
    }

    /**
     * Synthesizes audio from pre-phonemized segments (with silence gaps).
     * Use buildPhonemizedSegments() from utils.js to prepare the input.
     *
     * @param {{ ids: number[], silenceSamples: number }[]} segments
     * @param {number} [speakerId=0]
     * @param {object} [opts]
     * @returns {Promise<{ audioData: Float32Array, totalTimeMs: number, rtf: number }>}
     */
    async synthesizeFromSegments(segments, speakerId = 0, { noiseScale = 0.667, lengthScale = 1.0 } = {}) {
        const t0 = performance.now();
        const audioChunks = [];

        for (const seg of segments) {
            const segAudio = await this.synthesizeFromIds(seg.ids, speakerId, { noiseScale, lengthScale });
            if (segAudio.length > 0) audioChunks.push(segAudio);
            if (seg.silenceSamples > 0) audioChunks.push(new Float32Array(seg.silenceSamples));
        }

        const totalLength = audioChunks.reduce((acc, c) => acc + c.length, 0);
        const merged = new Float32Array(totalLength);
        let off = 0;
        for (const chunk of audioChunks) {
            merged.set(chunk, off);
            off += chunk.length;
        }

        const elapsed = performance.now() - t0;
        const durationS = totalLength / this.sampleRate;
        const rtf = durationS > 0 ? (elapsed / 1000) / durationS : 0;

        return { audioData: merged, totalTimeMs: Math.round(elapsed), rtf };
    }

    /**
     * High-level: synthesize text directly (requires phonemizer.js loaded).
     *
     * @param {string} text          Raw text to speak.
     * @param {number} [speakerId=0]
     * @param {object} [opts]
     * @returns {Promise<Float32Array>}
     */
    async synthesize(text, speakerId = 0, { noiseScale = 0.667, lengthScale = 1.0 } = {}) {
        const toSequence = resolveTextToSequence();

        // Apply comma padding (same as Python API) to prevent phonetic clipping
        let padded = text.trim();
        if (!padded.startsWith(',')) padded = ', ' + padded;
        if (!padded.endsWith(','))  padded = padded + ' ,';

        const ids = toSequence(padded);
        if (ids.length === 0) {
            throw new Error('Text produced no recognized phonemes.');
        }

        const t0 = performance.now();
        const audioData = await this.synthesizeFromIds(ids, speakerId, { noiseScale, lengthScale });
        const elapsedMs = performance.now() - t0;
        const durationS = audioData.length / this.sampleRate;
        const rtf = (elapsedMs / 1000) / durationS;

        console.log(`Synthesis: ${durationS.toFixed(2)}s of audio in ${elapsedMs.toFixed(0)}ms (RTF ${rtf.toFixed(3)})`);
        return audioData;
    }

    /**
     * Play audio directly via Web Audio API (no WAV file needed).
     * @param {Float32Array} floatData
     * @returns {Promise<AudioBufferSourceNode>}
     */
    async play(floatData) {
        if (!this.audioContext || this.audioContext.state === 'closed') {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        const ctx = this.audioContext;
        if (ctx.state === 'suspended') await ctx.resume();
        const buffer = ctx.createBuffer(1, floatData.length, this.sampleRate);
        buffer.copyToChannel(floatData, 0);
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        source.start();
        return source;
    }

    /** Release the reusable Web Audio context created by play(). */
    async closeAudio() {
        if (this.audioContext && this.audioContext.state !== 'closed') {
            await this.audioContext.close();
        }
        this.audioContext = null;
    }

    /**
     * Convert Float32Array audio to a downloadable WAV Blob (PCM16 mono).
     * @param {Float32Array} floatData
     * @returns {Blob}
     */
    toWavBlob(floatData) {
        return toWavBlob(floatData, this.sampleRate);
    }
}

/**
 * Encode Float32Array PCM audio into a WAV Blob.
 * @param {Float32Array} floatData
 * @param {number} [sampleRate=22050]
 * @returns {Blob}
 */
function toWavBlob(floatData, sampleRate = 22050) {
    const numSamples = floatData.length;
    const dataSize = numSamples * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);           // block align
    view.setUint16(34, 16, true);          // bits per sample
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, floatData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Split text by punctuation, phonemize each segment, and compute silence gaps.
 * Use with tts.synthesizeFromSegments() for natural pauses.
 *
 * @param {string} text
 * @returns {{ ids: number[], silenceSamples: number }[]}
 */
function buildPhonemizedSegments(text) {
    const toSequence = resolveTextToSequence();

    const rawSegments = text.split(/(?<=[,.;!?])\s+/);
    const segments = [];

    for (let i = 0; i < rawSegments.length; i++) {
        const segText = rawSegments[i].trim();
        if (!segText) continue;

        const ids = toSequence(segText);
        if (ids.length === 0) continue;

        const lastChar = segText[segText.length - 1];
        let silenceDuration = 0.15;
        if (['.', '!', '?'].includes(lastChar)) {
            silenceDuration = 0.45;
        } else if ([',', ';', ':'].includes(lastChar)) {
            silenceDuration = 0.2;
        }
        const isLast = i === rawSegments.length - 1;
        const silenceSamples = isLast ? 0 : Math.round(SAMPLE_RATE * silenceDuration);

        segments.push({ ids, silenceSamples });
    }

    return segments;
}

// Export for Node.js / CommonJS environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { KitsuneTTS, toWavBlob, buildPhonemizedSegments, readResponseBytes, SAMPLE_RATE };
}
