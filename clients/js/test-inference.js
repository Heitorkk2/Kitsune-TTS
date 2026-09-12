// SPDX-License-Identifier: GPL-3.0-only
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { KitsuneTTS } = require('./kitsune-tts.js');

test('stream yields before synthesizing the next segment and preserves silence', async () => {
    const tts = new KitsuneTTS();
    const calls = [];
    tts.synthesizeFromIds = async (ids, speaker, opts) => {
        calls.push({ ids, speaker, opts });
        return Float32Array.from(ids);
    };
    const segments = [{ ids: [1, 2], silenceSamples: 2 }, { ids: [3], silenceSamples: 0 }];
    const iterator = tts.streamFromSegments(segments, 4, { noiseScale: 0 });
    assert.equal(calls.length, 0);
    assert.deepEqual(Array.from((await iterator.next()).value), [1, 2]);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].speaker, 4);
    assert.equal(calls[0].opts.noiseScale, 0);
    assert.deepEqual(Array.from((await iterator.next()).value), [0, 0]);
    assert.equal(calls.length, 1);
    await iterator.return();
    assert.equal(calls.length, 1);
    const merged = await tts.synthesizeFromSegments(segments);
    assert.deepEqual(Array.from(merged.audioData), [1, 2, 0, 0, 3]);
    const empty = await tts.synthesizeFromSegments([]);
    assert.equal(empty.audioData.length, 0);
    assert.equal(empty.rtf, 0);
});

test('runtime tensors are disposed, but the returned PCM remains owned by the caller', async () => {
    for (const fail of [false, true]) {
        const tensors = [];
        class Tensor {
            constructor(type, data) { this.data = data; this.disposed = 0; tensors.push(this); }
            dispose() { this.disposed++; this.data.fill(this.data instanceof BigInt64Array ? 0n : 0); }
        }
        const tts = new KitsuneTTS();
        tts.ortApi = { Tensor };
        tts.session = { run: async () => {
            if (fail) throw new Error('inference failed');
            return { audio: new Tensor('float32', Float32Array.from([0.25, -0.5])) };
        } };
        if (fail) await assert.rejects(tts.synthesizeFromIds([1]), /inference failed/);
        else assert.deepEqual(Array.from(await tts.synthesizeFromIds([1])), [0.25, -0.5]);
        assert.ok(tensors.every(t => t.disposed === 1));
    }
});

test('stream propagates synthesis errors without starting later segments', async () => {
    const tts = new KitsuneTTS();
    let calls = 0;
    tts.synthesizeFromIds = async () => { calls++; throw new Error('failed'); };
    const iterator = tts.streamFromSegments([{ ids: [1] }, { ids: [2] }]);
    await assert.rejects(iterator.next(), /failed/);
    assert.equal(calls, 1);
});
