import os
import torch
import torch.utils.data
from .audio import load_wav, spectrogram_torch
from .symbols import cleaned_text_to_sequence
from ..phonemizer.phonemizer import EspeakPhonemizer

class KitsuneDataset(torch.utils.data.Dataset):
    """
    Loads audio, spectrograms, and text from a metadata file.
    Assumes 100% clean synthetic data (already sliced).
    Metadata format: audio_path|speaker_id|lang|text
    """
    def __init__(self, metadata_path, filter_length=1024, hop_length=256, win_length=1024, sampling_rate=22050):
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        
        self.audiopaths_and_text = self._load_metadata(metadata_path)
        self.phonemizer = EspeakPhonemizer() # Decoupled phonemizer
        
        # Validate sample rates of all files in metadata (fast check using wave module)
        import wave
        print("Validating audio sample rates...")
        for row in self.audiopaths_and_text:
            audiopath = row[0]
            if not os.path.exists(audiopath):
                raise FileNotFoundError(f"Audio file not found: {audiopath}")
            try:
                with wave.open(audiopath, 'rb') as f:
                    sr = f.getframerate()
                if sr != self.sampling_rate:
                    raise ValueError(f"Sample rate mismatch for {audiopath}: got {sr}Hz, expected {self.sampling_rate}Hz!")
            except Exception as e:
                raise ValueError(f"Error reading wave header for {audiopath}: {e}")
        
        # Pre-phonemize all texts to save time during __getitem__
        # (For huge datasets, we'd cache this to disk. Fine for RAM with 10-15h data).
        print("Pre-phonemizing dataset texts in memory...")
        self.phonemized_texts = []
        for row in self.audiopaths_and_text:
            # row: [audiopath, speaker_id, lang, text]
            phonemes = self.phonemizer.phonemize(row[3], lang=row[2])
            phoneme_ids = cleaned_text_to_sequence(phonemes)
            self.phonemized_texts.append(phoneme_ids)

    def _load_metadata(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Auto-detect delimiter
            delimiter = '\t' if '\t' in line else '|'
            parts = line.split(delimiter)
            
            # If it has 5 parts, assume the first is an index from Pandas/Excel and drop it
            if len(parts) >= 5:
                parts = parts[-4:] # take the last 4 parts (path, speaker, lang, text)
                
            if len(parts) == 4:
                data.append(parts)
        return data

    def __getitem__(self, index):
        audiopath, speaker_id, lang, raw_text = self.audiopaths_and_text[index]
        phoneme_ids = self.phonemized_texts[index]
        
        # Load Audio
        audio, sr = load_wav(audiopath)
        audio = torch.FloatTensor(audio)
        
        # Linear Spectrogram (for posterior encoder)
        spec = spectrogram_torch(audio.unsqueeze(0), self.filter_length,
                                 self.hop_length, self.win_length,
                                 center=False).squeeze(0)
        
        phoneme_tensor = torch.LongTensor(phoneme_ids)
        speaker_tensor = torch.LongTensor([int(speaker_id)])
        
        return (phoneme_tensor, spec, audio.unsqueeze(0), speaker_tensor)

    def __len__(self):
        return len(self.audiopaths_and_text)
