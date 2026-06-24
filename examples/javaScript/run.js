const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

// Import the Kitsune JS core
const { KitsuneTTS } = require('../../clients/js/kitsune-tts.js');
const { textToSequence } = require('../../clients/js/phonemizer.js');

// Mock browser globals for the kitsune-tts client to work in Node.js
global.ort = ort;
global.textToSequence = textToSequence;
global.performance = { now: () => Date.now() };

async function downloadModel(url, dest) {
    if (fs.existsSync(dest)) return;
    console.log(`Downloading ONNX model from HuggingFace to ${dest}...`);
    // Using native fetch available in Node.js 18+
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.statusText}`);
    const buffer = await res.arrayBuffer();
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, Buffer.from(buffer));
}

async function main() {
    console.log("🦊 Starting Kitsune-TTS (Node.js)");

    // Auto-download the ONNX model if missing
    const modelUrl = "https://huggingface.co/Heitorkk2/Kitsune-TTS-V1/resolve/main/kitsune39M.onnx";
    const modelPath = path.join(__dirname, 'kitsune39M.onnx');
    await downloadModel(modelUrl, modelPath);

    const tts = new KitsuneTTS();
    tts.ortApi = ort;
    
    console.log(`Loading model from: ${modelPath}`);
    
    // In Node, we can't easily fetch() local files for the TTS class, so we load the buffer directly
    const modelBuffer = fs.readFileSync(modelPath);
    tts.session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['cpu'],
        interOpNumThreads: 1
    });

    const text = "Olá! Este é um teste rodando diretamente no Node JS com download automático do modelo.";
    const speakerId = 0; // 0 = Emilia
    
    console.log(`Synthesizing text: "${text}"`);
    const audioFloat32 = await tts.synthesize(text, speakerId);
    
    console.log("Encoding to WAV...");
    const wavBlob = tts.toWavBlob(audioFloat32);
    
    // In Node 20+ Blob is native, we can extract the ArrayBuffer directly
    const arrayBuffer = await wavBlob.arrayBuffer();
    const outputPath = path.join(__dirname, 'output.wav');
    
    fs.writeFileSync(outputPath, Buffer.from(arrayBuffer));
    console.log(`✅ Saved to ${outputPath}!`);
}

main().catch(console.error);
