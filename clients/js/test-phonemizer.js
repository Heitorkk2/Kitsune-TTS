// SPDX-License-Identifier: GPL-3.0-only
'use strict';

const assert = require('node:assert/strict');
const {
    SYMBOL_TO_ID,
    markPrimaryStress,
    textToPhonemes,
    textToSequence,
    tokenizePhonemes,
} = require('./phonemizer.js');
const { buildPhonemizedSegments, readResponseBytes } = require('./kitsune-tts.js');

assert.equal(markPrimaryStress('rato'), 'rˈato');
assert.equal(markPrimaryStress('olá'), 'olˈá');
assert.equal(markPrimaryStress('coração'), 'coraçˈão');
assert.equal(markPrimaryStress('rei'), 'rˈei');
assert.equal(markPrimaryStress('roeu'), 'roˈeu');

const rato = textToPhonemes('O rato roeu.');
assert.match(rato, /xˈatʊ/);
assert.doesNotMatch(rato, /ʃˈatʊ/);
assert.match(rato, /\.$/);
assert.match(textToPhonemes('O rei roeu.'), /xˈeɪ xoˈeʊ/);
assert.match(textToPhonemes('Queijo.'), /kˈeɪ/);

const prosody = textToPhonemes('Olá, mundo!');
assert.match(prosody, /ˈ/);
assert.match(prosody, /,/);
assert.match(prosody, /!/);

assert.deepEqual(tokenizePhonemes('u\u0303'), ['ũ']);

const ids = textToSequence(', Olá, mundo! ,');
assert.ok(ids.length > 0);
assert.ok(ids.every(id => Number.isInteger(id) && id >= 0));
assert.ok(ids.includes(SYMBOL_TO_ID['ˈ']));
assert.ok(ids.includes(SYMBOL_TO_ID[',']));
assert.ok(ids.includes(SYMBOL_TO_ID['!']));

const segments = buildPhonemizedSegments('Olá, mundo! Tudo bem?');
assert.equal(segments.length, 3);
assert.ok(segments.every(segment => segment.ids.length > 0));
assert.ok(segments[0].silenceSamples > 0);
assert.equal(segments[2].silenceSamples, 0);

async function testStreamingDownload() {
    const chunks = [Uint8Array.from([1, 2, 3]), Uint8Array.from([4, 5, 6])];
    let index = 0;
    let lastProgress = null;
    const response = {
        headers: { get: name => name === 'content-length' ? '6' : null },
        body: {
            getReader: () => ({
                read: async () => index < chunks.length
                    ? { done: false, value: chunks[index++] }
                    : { done: true },
            }),
        },
        arrayBuffer: async () => { throw new Error('streaming path expected'); },
    };
    const bytes = await readResponseBytes(response, (...progress) => { lastProgress = progress; });
    assert.deepEqual(Array.from(bytes), [1, 2, 3, 4, 5, 6]);
    assert.deepEqual(lastProgress, [0, 0, 100]);
}

testStreamingDownload()
    .then(() => console.log('JavaScript client tests: OK'))
    .catch(error => {
        console.error(error);
        process.exitCode = 1;
    });
