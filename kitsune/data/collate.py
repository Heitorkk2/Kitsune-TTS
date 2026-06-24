import torch

class KitsuneCollate:
    """
    Zero-pads model inputs and targets based on number of frames per step.
    Orders batch by descending sequence length (for RNN efficiency if needed).
    """
    def __init__(self):
        pass

    def __call__(self, batch):
        """
        batch: list of tuples (phoneme_tensor, spec, audio, speaker_tensor)
        """
        # Order by spectrogram length (descending)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len).zero_()
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len).zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3][0]

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid
