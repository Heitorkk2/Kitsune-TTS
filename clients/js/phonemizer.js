/**
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Kitsune-TTS Client-Side Portuguese (PT-BR) Phonemizer and Text Normalizer.
 *
 * Rule-based fallback phonemizer that approximates espeak-ng IPA output.
 * Used when eSpeak WASM is unavailable (e.g. inside Web Workers).
 *
 * The normalizeText() pipeline mirrors the Python version in
 * kitsune/phonemizer/phonemizer.py — keep them in sync!
 */

const SYMBOLS = [
    '_',
    ';', ':', ',', '.', '!', '?', '¡', '¿', '—', '…', '"', '«', '»', '“', '”', ' ',
    'a', 'ã', 'e', 'ẽ', 'i', 'ĩ', 'o', 'õ', 'u', 'ũ', 'ɛ', 'ɔ', 'æ', 'œ', 'y', 'ø', 'ʌ', 'ɑ', 'ə', 'ɪ', 'ʊ', 'ɐ', 'ɚ', 'ɜ', 'ᵻ',
    'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 'ʃ', 'ʒ', 'ç', 'x', 'r', 'm', 'n', 'ɲ', 'ŋ', 'l', 'ʎ', 'j', 'h', 'w', 'θ', 'ð', 'ɹ', 'ɡ', 'ɾ',
    'ˈ', 'ˌ', 'ː', '̃'
];

const SYMBOL_TO_ID = {};
SYMBOLS.forEach((s, idx) => { SYMBOL_TO_ID[s] = idx; });
const SPACE_ID = SYMBOL_TO_ID[' '];

const VOWEL_RE = /[aãeẽiĩoõuũáéíóúâêô]/i;
const EXPLICIT_STRESS_RE = /[ãẽĩõũáéíóúâêô]/i;

// Special words with hand-tuned IPA — resolved BEFORE the general rules.
const SPECIAL_WORDS = {
    'emilia': 'ˌemˈiljæ',
    'frieren': 'frieɾeɪŋ',
    'violet': 'violɛtʃ',
    'evergarden': 'eveɾəɡaɾədeɪŋ',
    'espero': 'espɛɾʊ',
    'alegre': 'alɛɡry',
    'mais': 'maɪz',
    'você': 'vosˈe',
};

// Unstressed function words (clitics) — skip stress rules entirely.
const CLITICS = {
    'que': 'ky',
    'de': 'dʒy',
    'se': 'sy',
    'me': 'my',
    'te': 'tʃy',
    'o': 'ʊ',
    'a': 'a',
    'os': 'ʊs',
    'as': 'as',
    'um': 'ũŋ',
    'uma': 'umæ',
    'com': 'koŋ',
    'em': 'eɪŋ',
    'para': 'pæɾæ',
    'por': 'poɾ',
    'sem': 'seɪŋ',
    'do': 'dʊ',
    'da': 'da',
    'dos': 'dʊs',
    'das': 'das',
    'no': 'nʊ',
    'na': 'na',
    'nos': 'nʊs',
    'nas': 'nas',
    'ao': 'aʊ',
    'aos': 'aʊs',
    'ou': 'ow',
    'e': 'i',
};

function numToWordsPT(num) {
    const units = ["", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"];
    const teens = ["dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"];
    const tens = ["", "dez", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"];
    const hundreds = ["", "cem", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos", "setecentos", "oitocentos", "novecentos"];

    if (num === 0) return "zero";

    let parts = [];

    if (num >= 1000000) {
        let millions = Math.floor(num / 1000000);
        parts.push(millions === 1 ? "um milhão" : numToWordsPT(millions) + " milhões");
        num %= 1000000;
    }
    if (num >= 1000) {
        let thousands = Math.floor(num / 1000);
        parts.push(thousands === 1 ? "mil" : numToWordsPT(thousands) + " mil");
        num %= 1000;
    }
    if (num >= 100) {
        let h = Math.floor(num / 100);
        parts.push((h === 1 && num % 100 > 0) ? "cento" : hundreds[h]);
        num %= 100;
    }
    if (num >= 20) {
        let t = Math.floor(num / 10);
        let u = num % 10;
        parts.push(u > 0 ? tens[t] + " e " + units[u] : tens[t]);
    } else if (num >= 10) {
        parts.push(teens[num - 10]);
    } else if (num > 0) {
        parts.push(units[num]);
    }

    return parts.join(" e ");
}

/**
 * Normaliza texto bruto (datas, moeda, unidades, matemática, decimais...).
 */
function normalizeText(text) {
    let t = text.toLowerCase().trim();

    // 1. Datas: dd/mm/yyyy
    t = t.replace(/\b(\d{1,2})\/(\d{2})\/(\d{4})\b/g, (match, day, month, year) => {
        try {
            const d = parseInt(day, 10), m = parseInt(month, 10), y = parseInt(year, 10);
            const months = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
                "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"];
            const monthStr = months[m - 1] || month;
            return `${numToWordsPT(d)} de ${monthStr} de ${numToWordsPT(y)}`;
        } catch (e) { return match; }
    });

    // 2. Expoentes: 5² ou 10²
    t = t.replace(/\b(\d+)²/g, (match, base) => {
        try { return `${numToWordsPT(parseInt(base, 10))} ao quadrado`; }
        catch (e) { return match; }
    });

    // 2.5 Números negativos
    t = t.replace(/(?:^|\s)-(\d+(?:[.,]\d+)?)\b/g, (match, num) => match.replace('-' + num, ' menos ' + num));

    // 3. Horários: 17h42 / 23:59
    const timeReplacer = (match, h, m) => {
        try {
            const hr = parseInt(h, 10), min = parseInt(m, 10);
            const hrStr = numToWordsPT(hr);
            return min === 0 ? `${hrStr} horas` : `${hrStr} e ${numToWordsPT(min)}`;
        } catch (e) { return match; }
    };
    t = t.replace(/\b(\d{1,2})[hH](\d{2})\b/g, timeReplacer);
    t = t.replace(/\b(\d{1,2}):(\d{2})\b/g, timeReplacer);

    // 3.5. Graus e Celsius
    t = t.replace(/\b(\d+(?:[.,]\d+)?)\s*°c\b/g, '$1 graus celsius');
    t = t.replace(/\b(\d+(?:[.,]\d+)?)\s*°/g, '$1 graus ');

    // 4. Unidades
    const unitMap = {
        'km/h': ' quilômetros por hora', 'km²': ' quilômetros quadrados', 'mph': ' milhas por hora',
        'ghz': ' gigahertz', 'mhz': ' megahertz', 'hz': ' hertz', 'gbps': ' gigabits por segundo',
        'ml': ' mililitros', 'mm': ' milímetros', 'kg': ' quilogramas', 'km': ' quilômetros',
    };
    for (const [unit, val] of Object.entries(unitMap)) {
        const escaped = unit.replace(/[/\^$*+?.()|[\]{}]/g, '\\$&');
        t = t.replace(new RegExp('\\b(\\d+(?:[.,]\d+)?)\\s*' + escaped + '\\b', 'g'), '$1' + val);
        t = t.replace(new RegExp('\\b' + escaped + '\\b', 'g'), val.trim());
    }

    // 5. Separador de milhar: 1.234.567 -> 1234567
    t = t.replace(/\b(\d{1,3}(?:\.\d{3})+)\b/g, (match) => match.replace(/\./g, ''));

    // 6. Moeda: R$ 50,00 / $ 42 / € 10 / £ 5
    t = t.replace(/(r\$|\$|€|£)\s*(\d+(?:[.,]\d{1,2})?)/g, (match, sym, valStr) => {
        try {
            const val = parseFloat(valStr.replace(',', '.'));
            const intPart = Math.floor(val);
            const cents = Math.round((val - intPart) * 100);
            const symMap = { 'r$': 'real', '$': 'dólar', '€': 'euro', '£': 'libra' };
            const symPlural = { 'r$': 'reais', '$': 'dólares', '€': 'euros', '£': 'libras' };
            const currName = intPart === 1 ? symMap[sym] : symPlural[sym];
            const intStr = numToWordsPT(intPart);
            if (cents === 0) return `${intStr} ${currName}`;
            const centName = cents === 1 ? 'centavo' : 'centavos';
            return `${intStr} ${currName} e ${numToWordsPT(cents)} ${centName}`;
        } catch (e) { return match; }
    });

    // 6.5. Phone Numbers / Postal Codes (e.g. 98765-4321, (11) 98765-4321)
    t = t.replace(/(?:\(?\d{2}\)?\s*)?\b\d{4,5}-\d{3,4}\b/g, (match) => {
        const digits = match.replace(/\D/g, '');
        return digits.split('').map(d => numToWordsPT(parseInt(d, 10))).join(' ');
    });

    // 7. Decimals: 3,14 -> três vírgula um quatro
    t = t.replace(/\b(\d+)[,.](\d+)\b/g, (match, intPart, decDigits) => {
        try {
            const intStr = numToWordsPT(parseInt(intPart, 10));
            const decStr = decDigits.split('').map(d => numToWordsPT(parseInt(d, 10))).join(' ');
            return `${intStr} vírgula ${decStr}`;
        } catch (e) { return match; }
    });

    // 8. Operadores matemáticos colados a dígitos
    t = t.replace(/(\d+)\s*\+\s*(\d+)/g, '$1 mais $2');
    t = t.replace(/(\d+)\s*-\s*(\d+)/g, '$1 menos $2');
    t = t.replace(/(\d+)\s*[xX*×]/g, '$1 vezes');
    t = t.replace(/(\d+)\s*[/÷]/g, '$1 dividido por');
    t = t.replace(/(\d+)\s*=/g, '$1 é igual a');
    const mathMap = {
        ' + ': ' mais ', ' - ': ' menos ', ' × ': ' vezes ', ' * ': ' vezes ',
        ' = ': ' é igual a ', ' ÷ ': ' dividido por ', ' / ': ' dividido por ', '%': ' por cento',
    };
    for (const [op, val] of Object.entries(mathMap)) { t = t.replaceAll(op, val); }

    // 9. Números isolados
    t = t.replace(/\b\d+\b/g, (match) => numToWordsPT(parseInt(match, 10)));

    // Limpa símbolos soltos remanescentes
    t = t.replace(/ \$ /g, ' ').replace(/ r\$ /g, ' ');

    // 10. Operadores soltos remanescentes
    t = t.replace(/\s+-\s+/g, ' menos ');
    t = t.replace(/\s+=\s+/g, ' é igual a ');

    t = t.replace(/\s+/g, " ");
    return t;
}

/**
 * Inserts a primary-stress marker using the default PT-BR orthographic rules.
 * Hand-tuned words and clitics bypass this function. The marker is inserted
 * before grapheme-to-phoneme substitutions, so it follows the stressed vowel
 * through the rule pipeline.
 */
function markPrimaryStress(word) {
    const chars = Array.from(word);
    const vowelIndices = [];
    const diphthongs = new Set(['ai', 'ei', 'oi', 'ui', 'eu', 'ou', 'au', 'iu']);

    for (let i = 0; i < chars.length; i++) {
        if (!VOWEL_RE.test(chars[i])) continue;
        vowelIndices.push(i);
        const pair = (chars[i] + (chars[i + 1] || '')).toLowerCase();
        if (diphthongs.has(pair)) i += 1;
    }
    if (vowelIndices.length === 0) return word;

    let stressIndex = vowelIndices.find(i => EXPLICIT_STRESS_RE.test(chars[i]));
    if (stressIndex === undefined) {
        // Words ending in a/e/o, a nasal ending, or s normally stress the
        // penultimate vowel nucleus; other endings normally stress the last.
        const penultimateEnding = /(?:[aeo]|[aeo]s|am|em|ens)$/i.test(word);
        const vowelPosition = penultimateEnding && vowelIndices.length > 1
            ? vowelIndices.length - 2
            : vowelIndices.length - 1;
        stressIndex = vowelIndices[vowelPosition];
    }

    chars.splice(stressIndex, 0, 'ˈ');
    return chars.join('');
}

/**
 * Converts PT-BR text into approximate IPA phonemes (espeak-ng compatible).
 */
function textToPhonemes(text) {
    let normalized = normalizeText(text);
    const words = normalized.split(' ');

    const phonemizedWords = words.map(word => {
        if (!word) return '';
        
        // Separa prefixos e sufixos de pontuação para o phonemizer rodar na palavra limpa
        const match = word.match(/^([^a-zãẽĩõũçüáéíóúâêô]*)(.*?)([^a-zãẽĩõũçüáéíóúâêô]*)$/i);
        if (!match) return word;
        const prefix = match[1];
        const cleanWord = match[2];
        const suffix = match[3];

        if (!cleanWord) return word;

        // Nomes especiais ANTES de aplicar as regras
        const lower = cleanWord.toLowerCase();
        if (SPECIAL_WORDS[lower]) return prefix + SPECIAL_WORDS[lower] + suffix;

        // Palavras funcionais átonas (clíticos) pulam acentuação tônica e regras gerais
        if (CLITICS[lower]) return prefix + CLITICS[lower] + suffix;

        let w = markPrimaryStress(cleanWord);

        // Silent 'u' in 'qu' and 'gu' before e/i (needs to run first so the 'u' isn't treated as a vowel/diphthong)
        w = w.replace(/qu(?=ˈ?[eiíéê])/g, 'k');
        w = w.replace(/gu(?=ˈ?[eiíéê])/g, 'ɡ');

        // Ditongos e vogais especiais
        w = w.replace(/eu$/g, 'eʊ');
        w = w.replace(/eu(?=[^aeiou|áéíóúâêôãõ])/g, 'eʊ');
        w = w.replace(/^sou$/g, 'sow');
        w = w.replace(/ou$/g, 'ow');
        w = w.replace(/ou/g, 'ow');
        w = w.replace(/ei/g, 'eɪ');
        w = w.replace(/ai/g, 'aɪ');
        w = w.replace(/ui/g, 'uɪ');

        // Palatalização de "te"/"de" final
        w = w.replace(/te$/g, 'tʃy');
        w = w.replace(/de$/g, 'dʒy');
        w = w.replace(/tes$/g, 'tʃys');
        w = w.replace(/des$/g, 'dʒys');

        // Consoantes duplas e dígrafos
        w = w.replace(/lh/g, 'ʎ');
        w = w.replace(/nh/g, 'ɲ');
        w = w.replace(/ch/g, 'ʃ');
        // Convert orthographic x first. Otherwise the /r/ allophone marker
        // introduced below would immediately be converted to /ʃ/ as well.
        w = w.replace(/x/g, 'ʃ');
        w = w.replace(/rr/g, 'x');
        w = w.replace(/^r/g, 'x');

        // Tepe alveolar (ɾ) entre vogais
        w = w.replace(/([aeiouãõáéíóúâêô])r(ˈ?[aeiouãõáéíóúâêô])/g, '$1\u027E$2');

        // R final de sílaba antes de consoante e no final do termo
        w = w.replace(/([aeiouãõáéíóúâêô])r(?=[pbtdkgfvszʃʒçxmnɲŋʎjhwðɹɡ])/g, '$1\u027Eə');
        w = w.replace(/ar$/g, 'aɾ');
        w = w.replace(/er$/g, 'er');
        w = w.replace(/ir$/g, 'iɾ');
        w = w.replace(/or$/g, 'or');
        w = w.replace(/ur$/g, 'ur');

        // Nasalização
        w = w.replace(/a[mn](?=[pb])/g, 'ɐ\u0303m');
        w = w.replace(/a[mn]$/g, 'ɐ\u0303ŋ');
        w = w.replace(/a[mn](?=[^aeiouáéíóúâêô])/g, 'ɐ\u0303ŋ');
        w = w.replace(/e[mn]$/g, 'eɪŋ');
        w = w.replace(/e[mn](?=[^aeiouáéíóúâêô])/g, 'eɪŋ');
        w = w.replace(/i[mn]$/g, 'i\u0303ŋ');
        w = w.replace(/i[mn](?=[^aeiouáéíóúâêô])/g, 'i\u0303ŋ');
        w = w.replace(/o[mn]$/g, 'oŋ');
        w = w.replace(/o[mn](?=[^aeiouáéíóúâêô])/g, 'oŋ');
        w = w.replace(/u[mn]$/g, 'u\u0303ŋ');
        w = w.replace(/u[mn](?=[^aeiouáéíóúâêô])/g, 'u\u0303ŋ');

        // Consoantes
        w = w.replace(/([aeiouãõáéíóúâêô])s(ˈ?[aeiouãõáéíóúâêô])/g, '$1z$2');
        w = w.replace(/ss/g, 's');
        w = w.replace(/c(?=ˈ?[eiíéê])/g, 's');
        w = w.replace(/c(?=ˈ?[aouáóúâôãõ])/g, 'k');
        w = w.replace(/c$/g, 'k');
        w = w.replace(/ç/g, 's');
        w = w.replace(/g(?=ˈ?[eiíéê])/g, 'ʒ');
        w = w.replace(/g(?=ˈ?[aouáóúâôãõ])/g, 'ɡ');
        w = w.replace(/j/g, 'ʒ');

        // Palatalização do D e T antes de i/y (ex: "dia" -> "dʒia")
        w = w.replace(/d(?=ˈ?[iyíýĩ])/g, 'dʒ');
        w = w.replace(/t(?=ˈ?[iyíýĩ])/g, 'tʃ');

        // Vogais finais átonas
        w = w.replace(/o$/g, 'ʊ');
        w = w.replace(/e$/g, 'y');
        w = w.replace(/a$/g, 'æ');

        // Vogais acentuadas -> padrão
        w = w.replace(/[áâ]/g, 'a');
        w = w.replace(/é/g, 'ɛ');
        w = w.replace(/ê/g, 'e');
        w = w.replace(/[íî]/g, 'i');
        w = w.replace(/ó/g, 'ɔ');
        w = w.replace(/ô/g, 'o');
        w = w.replace(/[úû]/g, 'u');

        // L vocálico -> W
        w = w.replace(/l$/g, 'w');
        w = w.replace(/l(?=[pbtdkgfvszʃʒçxmnɲŋʎjhwθðɹɡ])/g, 'w');
        w = w.replace(/l/g, 'l');

        return prefix + w + suffix;
    });

    let w = phonemizedWords.join(' ');

    // Keep punctuation and prosodic modifiers: they are explicit model tokens
    // and are preserved by the Python/eSpeak training pipeline.
    w = w.replace(/[^a-zãẽĩõũɛɔæœyøʌɑəɪʊɐɚɜᵻpbtdkgfvszʃʒçxrmnɲŋlʎjhwθðɹɡɾˈˌː̃ ;:,.!?¡¿—…"«»“”]/g, '');
    w = w.replace(/\s+/g, ' ').trim().normalize('NFC');

    return w;
}

function tokenizePhonemes(phonemeStr) {
    const tokens = [];
    phonemeStr = phonemeStr.normalize('NFC');
    let i = 0;
    while (i < phonemeStr.length) {
        if (i + 1 < phonemeStr.length) {
            let twoChars = phonemeStr.substr(i, 2);
            if (SYMBOL_TO_ID[twoChars] !== undefined) {
                tokens.push(twoChars);
                i += 2;
                continue;
            }
        }
        let oneChar = phonemeStr.charAt(i);
        if (SYMBOL_TO_ID[oneChar] !== undefined) {
            tokens.push(oneChar);
        } // else: silently drop unknown symbols
        i += 1;
    }
    return tokens;
}

function textToSequence(text) {
    const phonemes = textToPhonemes(text);
    const tokens = tokenizePhonemes(phonemes);
    const sequence = [];
    tokens.forEach(tok => {
        if (SYMBOL_TO_ID[tok] !== undefined) sequence.push(SYMBOL_TO_ID[tok]);
    });
    return sequence;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        normalizeText,
        markPrimaryStress,
        textToPhonemes,
        tokenizePhonemes,
        textToSequence,
        SYMBOLS,
        SYMBOL_TO_ID,
        SPACE_ID,
    };
}
