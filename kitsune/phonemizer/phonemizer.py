from abc import ABC, abstractmethod
import re
import unicodedata

try:
    from num2words import num2words
except ImportError:
    num2words = None

class BasePhonemizer(ABC):
    """
    Abstract base class for the Kitsune Phonemizer.
    This design isolates the phonemizer backend so we can swap it out later
    (e.g. replacing a GPL espeak-ng backend with a custom MIT-licensed dictionary).
    """

    @abstractmethod
    def phonemize(self, text: str, lang: str = "pt-br") -> list[str]:
        """
        Convert raw text into a list of phoneme strings.
        
        Args:
            text: Raw input string.
            lang: Target language code (e.g. 'pt-br', 'en', 'ja').
            
        Returns:
            A list of phonemes representing the text.
        """
        pass

    def get_supported_languages(self) -> list[str]:
        """Return a list of language codes supported by this phonemizer."""
        return ["pt-br", "en", "ja"]

    def _normalize_text(self, text: str, lang: str = "pt-br") -> str:
        """
        Basic normalization to clean up the string before phonemization.
        Subclasses can extend this as needed.
        """
        # Lowercase, clean up extra spaces
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Expand numbers using num2words if available
        if num2words is not None:
            # Map lang to num2words lang code
            n2w_lang = {"pt-br": "pt_BR", "en": "en", "ja": "ja"}.get(lang, "en")
            
            # 1. Dates: dd/mm/yyyy -> e.g. 14/03/2028
            if lang == "pt-br":
                def replace_date(match):
                    try:
                        day = int(match.group(1))
                        month = int(match.group(2))
                        year = int(match.group(3))
                        months = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", 
                                  "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
                        month_str = months[month - 1] if 1 <= month <= 12 else str(month)
                        day_str = num2words(day, lang='pt_BR')
                        year_str = num2words(year, lang='pt_BR')
                        return f"{day_str} de {month_str} de {year_str}"
                    except Exception:
                        return match.group(0)
                text = re.sub(r'\b(\d{1,2})/(\d{2})/(\d{4})\b', replace_date, text)
                
            # 2. Exponents: 5² or 10²
            def replace_exponent(match):
                try:
                    base = int(match.group(1))
                    base_str = num2words(base, lang=n2w_lang)
                    suffix = " ao quadrado" if lang == "pt-br" else " squared"
                    return f"{base_str}{suffix}"
                except Exception:
                    return match.group(0)
            text = re.sub(r'\b(\d+)²', replace_exponent, text)

            # 2.5. Negative numbers: -60 or -3,14
            if lang == "pt-br":
                text = re.sub(r'(?:^|\s)-(\d+(?:[.,]\d+)?)\b', r' menos \1', text)
            else:
                text = re.sub(r'(?:^|\s)-(\d+(?:[.,]\d+)?)\b', r' minus \1', text)

            # 3. Time format: 17h42, 19h30, 23h59
            if lang == "pt-br":
                def replace_time_h(match):
                    try:
                        h = int(match.group(1))
                        m = int(match.group(2))
                        h_str = num2words(h, lang='pt_BR')
                        if m == 0:
                            return f"{h_str} horas"
                        m_str = num2words(m, lang='pt_BR')
                        return f"{h_str} e {m_str}" # e.g. dezoito e trinta
                    except Exception:
                        return match.group(0)
                text = re.sub(r'\b(\d{1,2})[hH](\d{2})\b', replace_time_h, text)
                
                # 23:59 -> vinte e três e cinquenta e nove
                def replace_time_colon(match):
                    try:
                        h = int(match.group(1))
                        m = int(match.group(2))
                        h_str = num2words(h, lang='pt_BR')
                        if m == 0:
                            return f"{h_str} horas"
                        m_str = num2words(m, lang='pt_BR')
                        return f"{h_str} e {m_str}"
                    except Exception:
                        return match.group(0)
                text = re.sub(r'\b(\d{1,2}):(\d{2})\b', replace_time_colon, text)

            # 3.5. Degrees and Celsius: 37°C or 38°
            if lang == "pt-br":
                text = re.sub(r'\b(\d+(?:[.,]\d+)?)\s*°c\b', r'\1 graus celsius', text)
                text = re.sub(r'\b(\d+(?:[.,]\d+)?)\s*°', r'\1 graus ', text)
            else:
                text = re.sub(r'\b(\d+(?:[.,]\d+)?)\s*°c\b', r'\1 degrees celsius', text)
                text = re.sub(r'\b(\d+(?:[.,]\d+)?)\s*°', r'\1 degrees ', text)

            # 4. Units expansion (Move before separators so space is added, allowing word boundaries to match)
            if lang == "pt-br":
                unit_map = {
                    'km/h': ' quilômetros por hora',
                    'km²': ' quilômetros quadrados',
                    'mph': ' milhas por hora',
                    'ghz': ' gigahertz',
                    'mhz': ' megahertz',
                    'hz': ' hertz',
                    'gbps': ' gigabits por segundo',
                    'ml': ' mililitros',
                    'mm': ' milímetros',
                    'kg': ' quilogramas',
                    'km': ' quilômetros',
                }
            else:
                unit_map = {
                    'km/h': ' kilometers per hour',
                    'km²': ' square kilometers',
                    'mph': ' miles per hour',
                    'ghz': ' gigahertz',
                    'mhz': ' megahertz',
                    'hz': ' hertz',
                    'gbps': ' gigabits per second',
                    'ml': ' milliliters',
                    'mm': ' millimeters',
                    'kg': ' kilograms',
                    'km': ' kilometers',
                }
            for unit, val in unit_map.items():
                # 1. Replace when preceded by a number (and optional spaces)
                text = re.sub(r'\b(\d+(?:[.,]\d+)?)\s*' + re.escape(unit) + r'(?![a-zA-Z0-9_])', r'\1' + val, text)
                # 2. Also replace when standalone (preceded by a word boundary or space, not a number)
                text = re.sub(r'\b' + re.escape(unit) + r'(?![a-zA-Z0-9_])', val.strip(), text)

            # 5. Thousands separators cleanup (handles chained groups, e.g. 1.234.567)
            if lang == "pt-br":
                text = re.sub(r'\b(\d{1,3}(?:\.\d{3})+)\b',
                              lambda m: m.group(1).replace('.', ''), text)
            else:
                text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)\b',
                              lambda m: m.group(1).replace(',', ''), text)

            # 6. Currencies with symbols (e.g. r$ 50,00)
            def replace_currency(match):
                try:
                    sym = match.group(1).strip()
                    val = float(match.group(2).replace(',', '.'))
                    
                    curr_map = {'r$': 'BRL', '$': 'USD', '€': 'EUR', '£': 'GBP'}
                    curr_code = curr_map.get(sym, 'USD')
                    
                    kwargs = {}
                    if n2w_lang != "pt_BR":
                        kwargs['currency'] = curr_code
                        
                    res = num2words(val, to='currency', lang=n2w_lang, **kwargs)
                    if val.is_integer():
                        if lang == "pt-br":
                            res = res.replace(" e zero centavos", "")
                        else:
                            res = res.replace(", zero cents", "").replace(", zero pence", "")
                    return res
                except Exception as e:
                    return match.group(0)
            text = re.sub(r'(r\$|\$|€|£)\s*(\d+(?:[.,]\d{1,2})?)', replace_currency, text)
            
            # 6.5. Phone Numbers / Postal Codes (e.g. 98765-4321, 12345-678, (11) 98765-4321)
            def replace_phone(match):
                try:
                    # Strip everything but digits
                    digits = re.sub(r'\D', '', match.group(0))
                    # Read digit by digit
                    words = [num2words(int(d), lang=n2w_lang) for d in digits]
                    return " ".join(words)
                except Exception:
                    return match.group(0)
            
            # Matches optional (XX) area code, followed by 4 or 5 digits, a hyphen, and 3 or 4 digits
            text = re.sub(r'(?:\(?\d{2}\)?\s*)?\b\d{4,5}-\d{3,4}\b', replace_phone, text)
            
            # 7. Decimals (e.g. 3,14 or 2.1). The fractional part is read
            # digit-by-digit ("três vírgula um quatro"), which is how decimals
            # are actually spoken — not as a cardinal ("mil quatrocentos").
            def replace_decimal(match):
                try:
                    int_part = num2words(int(match.group(1)), lang=n2w_lang)
                    dec_digits = match.group(2)
                    dec_part = " ".join(
                        num2words(int(d), lang=n2w_lang) for d in dec_digits
                    )
                    sep = " vírgula " if lang == "pt-br" else " point "
                    return f"{int_part}{sep}{dec_part}"
                except Exception:
                    return match.group(0)
            if lang == "pt-br":
                text = re.sub(r'\b(\d+),(\d+)\b', replace_decimal, text)
                text = re.sub(r'\b(\d+)\.(\d{1,2})\b', replace_decimal, text)
            else:
                text = re.sub(r'\b(\d+)\.(\d+)\b', replace_decimal, text)
            
            # 8. Math operators
            if lang == "pt-br":
                # Compact digit math (like 2+2 or 5-3)
                text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 mais \2', text)
                text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 menos \2', text)
                text = re.sub(r'(\d+)\s*[xX*×]\s*(\d+)', r'\1 vezes \2', text)
                text = re.sub(r'(\d+)\s*[/÷]\s*(\d+)', r'\1 dividido por \2', text)
                text = re.sub(r'(\d+)\s*=\s*(\d+)', r'\1 é igual a \2', text)
                
                math_map = {
                    ' + ': ' mais ',
                    ' - ': ' menos ',
                    ' × ': ' vezes ',
                    ' * ': ' vezes ',
                    ' = ': ' é igual a ',
                    ' ÷ ': ' dividido por ',
                    ' / ': ' dividido por ',
                    '%': ' por cento',
                }
            else:
                # Compact digit math (like 2+2 or 5-3)
                text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 plus \2', text)
                text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 minus \2', text)
                text = re.sub(r'(\d+)\s*[xX*×]\s*(\d+)', r'\1 times \2', text)
                text = re.sub(r'(\d+)\s*[/÷]\s*(\d+)', r'\1 divided by \2', text)
                text = re.sub(r'(\d+)\s*=\s*(\d+)', r'\1 equals \2', text)
                
                math_map = {
                    ' + ': ' plus ',
                    ' - ': ' minus ',
                    ' × ': ' times ',
                    ' * ': ' times ',
                    ' = ': ' equals ',
                    ' ÷ ': ' divided by ',
                    ' / ': ' divided by ',
                    '%': ' percent',
                }
            for op, val in math_map.items():
                text = text.replace(op, val)

            # 9. Isolated numbers
            def replace_num(match):
                try:
                    return num2words(int(match.group(0)), lang=n2w_lang)
                except Exception:
                    return match.group(0)
                    
            text = re.sub(r'\b\d+\b', replace_num, text)
            
            # Clean up dangling symbols (like if the user wrote "50 $ reais")
            text = text.replace(' $ ', ' ').replace(' r$ ', ' ')
            
            # 10. Final loose math operators cleanup (handles cases like word - word or word = word)
            if lang == "pt-br":
                text = re.sub(r'\s+-\s+', ' menos ', text)
                text = re.sub(r'\s+=\s+', ' é igual a ', text)
            else:
                text = re.sub(r'\s+-\s+', ' minus ', text)
                text = re.sub(r'\s+=\s+', ' equals ', text)
            
        return text


class EspeakPhonemizer(BasePhonemizer):
    """
    Prototype implementation using espeak-ng.
    Note: Requires phonemizer library (`pip install phonemizer`) and espeak-ng installed on the system.
    """
    
    # Word-boundary token kept between words so the model can learn
    # segmentation (the symbol table reserves ' ' / SPACE_ID for this).
    WORD_BOUNDARY = " "
    ESPEAK_LANGUAGES = {"pt-br": "pt-br", "en": "en-us", "ja": "ja"}

    def __init__(
        self,
        preserve_punctuation: bool = True,
        with_stress: bool = True,
        eager_languages=("pt-br", "en"),
    ):
        import os
        import sys
        
        # Windows-specific fallback for eSpeak-NG
        if sys.platform == "win32":
            espeak_exe = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
            espeak_dll = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
            if 'PHONEMIZER_ESPEAK_PATH' not in os.environ and os.path.exists(espeak_exe):
                os.environ['PHONEMIZER_ESPEAK_PATH'] = espeak_exe
            if 'PHONEMIZER_ESPEAK_LIBRARY' not in os.environ and os.path.exists(espeak_dll):
                os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = espeak_dll

        try:
            from phonemizer.backend import EspeakBackend
        except ImportError:
            raise ImportError("Please install phonemizer package: pip install phonemizer")

        self._backend_class = EspeakBackend
        self._backend_options = {
            "language_switch": "remove-flags",
            "preserve_punctuation": preserve_punctuation,
            "with_stress": with_stress,
        }
        self.backends = {}

        # Japanese is commonly unavailable in desktop eSpeak installations, so
        # callers can keep it lazy while preserving eager PT-BR/English behavior.
        for lang_code in eager_languages:
            self._load_backend(lang_code, warn=False)

    def _load_backend(self, lang_code: str, warn: bool = True):
        if lang_code in self.backends:
            return self.backends[lang_code]
        espeak_lang = self.ESPEAK_LANGUAGES.get(lang_code)
        if espeak_lang is None:
            return None
        try:
            backend = self._backend_class(espeak_lang, **self._backend_options)
            self.backends[lang_code] = backend
            return backend
        except Exception as error:
            if warn:
                print(f"[Warning] Could not load espeak backend for '{lang_code}': {error}")
            return None

    def get_supported_languages(self) -> list[str]:
        """Only the languages whose espeak backend actually loaded."""
        return list(self.backends.keys())

    def _tokenize_phonemes(self, phonemes_str: str) -> list[str]:
        """
        Split an espeak IPA string into phoneme tokens aligned with the
        single-codepoint symbol table.

        NFC normalization collapses espeak's decomposed output (base + combining
        mark) onto the precomposed vowels the table defines (e.g. u + U+0303 ->
        U+0169). Nasal vowels with no precomposed form (e.g. the vowel in "-ao")
        stay as base + standalone tilde -- both are in the table. Word
        boundaries are preserved as a single WORD_BOUNDARY token; unknown
        symbols (tie bars, etc.) are dropped later by cleaned_text_to_sequence.
        """
        phonemes_str = unicodedata.normalize("NFC", phonemes_str)
        tokens: list[str] = []
        for ch in phonemes_str:
            if ch.isspace():
                if tokens and tokens[-1] != self.WORD_BOUNDARY:
                    tokens.append(self.WORD_BOUNDARY)
            else:
                tokens.append(ch)

        while tokens and tokens[0] == self.WORD_BOUNDARY:
            tokens.pop(0)
        while tokens and tokens[-1] == self.WORD_BOUNDARY:
            tokens.pop()
        return tokens

    def phonemize(self, text: str, lang: str = "pt-br") -> list[str]:
        text = self._normalize_text(text, lang)

        backend = self._load_backend(lang)
        if backend is None:
            if not self.backends and self._load_backend("pt-br") is None:
                raise RuntimeError(
                    "No espeak backends loaded; check that espeak-ng is installed."
                )
            backend = self.backends.get("pt-br") or next(iter(self.backends.values()))

        # Keep word boundaries (strip=False) so we can emit a boundary token.
        phonemes_str = backend.phonemize([text], strip=False)[0]

        return self._tokenize_phonemes(phonemes_str)
