"""
Defines the custom reduced phoneme symbol table for Kitsune-TTS.
Instead of using the full 150+ IPA set from espeak, we use a curated
subset of ~75 symbols focusing on PT-BR, with enough coverage for
"Brazilianized" English and Japanese Romaji.

This keeps the embedding table small and inference fast.
"""

_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '

# Curated Vowels (Oral + Nasal + English)
_vowels = 'aãeẽiĩoõuũɛɔæœyøʌɑəɪʊɐɚɜᵻ' 

# Curated Consonants (+ English + PT-BR flap)
_consonants = 'pbtdkgfvszʃʒçxrmnɲŋlʎjhwθðɹɡɾ'

# Modifiers (Primary stress, secondary stress, length mark, nasal)
_modifiers = 'ˈˌː̃'

# Combine all symbols
symbols = [_pad] + list(_punctuation) + list(_vowels) + list(_consonants) + list(_modifiers)

# Dictionaries for O(1) lookup
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

SPACE_ID = _symbol_to_id[' ']

def cleaned_text_to_sequence(text):
    """Converts a string of phonemes to a sequence of IDs."""
    sequence = []
    for symbol in text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
    return sequence

def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string."""
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]
    return result
