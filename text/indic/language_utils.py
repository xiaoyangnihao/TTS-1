"""Indic language utilities"""

# Unicode offset to ITRANS mapping
offset_itrans_mapping = {
    0x2: ".m",
    0x3: "H",
    0x5: "a",
    0x6: "aa",
    0x7: "i",
    0x8: "ii",
    0x9: "u",
    0xa: "uu",
    0xb: "R^i",
    0xc: "L^i",
    0xe: ".e",
    0xf: "e",
    0x10: "ai",
    0x12: ".o",
    0x13: "o",
    0x14: "au",
    0x15: "ka",
    0x16: "kha",
    0x17: "ga",
    0x18: "gha",
    0x19: "~Na",
    0x1a: "cha",
    0x1b: "Cha",
    0x1c: "ja",
    0x1d: "jha",
    0x1e: "~na",
    0x1f: "Ta",
    0x20: "Tha",
    0x21: "Da",
    0x22: "Dha",
    0x23: "Na",
    0x24: "ta",
    0x25: "tha",
    0x26: "da",
    0x27: "dha",
    0x28: "na",
    0x29: "*na",
    0x2a: "pa",
    0x2b: "pha",
    0x2c: "ba",
    0x2d: "bha",
    0x2e: "ma",
    0x2f: "ya",
    0x30: "ra",
    0x31: "Ra",
    0x32: "la",
    0x33: "lda",
    0x34: "zha",
    0x35: "va",
    0x36: "sha",
    0x37: "Sha",
    0x38: "sa",
    0x39: "ha",
    0x3d: ".a",
    0x3e: "aa",
    0x3f: "i",
    0x40: "ii",
    0x41: "u",
    0x42: "uu",
    0x43: "R^i",
    0x44: "R^I",
    0x46: ".e",
    0x47: "e",
    0x48: "ai",
    0x4a: ".o",
    0x4b: "o",
    0x4c: "au",
    0x4d: "",
    0x50: "AUM",
    0x60: "R^I",
    0x61: "L^I",
    0x62: "L^i",
    0x63: "L^I",
    0x64: ".",
    0x65: "..",
    0x66: "0",
    0x67: "1",
    0x68: "2",
    0x69: "3",
    0x6a: "4",
    0x6b: "5",
    0x6c: "6",
    0x6d: "7",
    0x6e: "8",
    0x6f: "9",
    0x70: "॰",
    0x71: "ॱ",
    0x7f: "a",
}

# Unicode codepoint ranges for Indic languages
language_unicode_range_mapping = {
    'ta': [0x0b80, 0x0bff],
    'te': [0x0c00, 0x0c7f],
    'ml': [0x0d00, 0x0d7f],
    'hi': [0x0900, 0x097f],
    'bn': [0x0980, 0x09ff],
    'as': [0x0980, 0x09ff],
}

# Common offsets
COORDINATED_RANGE_START_INCLUSIVE = 0
COORDINATED_RANGE_END_INCLUSIVE = 0x6f

NUMERIC_OFFSET_START = 0x66
NUMERIC_OFFSET_END = 0x6f

HALANTA_OFFSET = 0x4d
AUM_OFFSET = 0x50
NUKTA_OFFSET = 0x3c

# Common symbols
RUPEE_SIGN = 0x20b9
DANDA = 0x0964
DOUBLE_DANDA = 0x0965


def char_to_offset(char, lang_code):
    """Get unicode offset corresponding of a character in a particular language
    """
    return ord(char) - language_unicode_range_mapping[lang_code][0]


def offset_to_char(offset, lang_code):
    """Get character in a particular language corresponding to an unicode offset
    """
    return chr(offset + language_unicode_range_mapping[lang_code][0])


def is_vowel_sign_offset(offset):
    """Returns if the offset is a vowel sign (maatraa)
    """
    return (offset >= 0x3e and offset <= 0x4c)


def is_halanta_offset(offset):
    """Returns if the offset is a halantha
    """
    return (offset == HALANTA_OFFSET)


def unicode_to_itrans_tranliteration(text, lang_code):
    """Transliterate Unicode text in a particular Indic language to ITRANS
    This effectively performs romanization of Indic text
    """
    if lang_code in language_unicode_range_mapping:
        if lang_code == "ml":
            # Change from chillus characters to corresponding consonant+halant
            text = text.replace('\u0d7a', '\u0d23\u0d4d')
            text = text.replace('\u0d7b', '\u0d28\u0d4d')
            text = text.replace('\u0d7c', '\u0d30\u0d4d')
            text = text.replace('\u0d7d', '\u0d32\u0d4d')
            text = text.replace('\u0d7e', '\u0d33\u0d4d')
            text = text.replace('\u0d7f', '\u0d15\u0d4d')

        offsets = [char_to_offset(char, lang_code) for char in text]

        itrans_text = []
        for o in offsets:
            itrans = offset_itrans_mapping.get(
                o, chr(language_unicode_range_mapping[lang_code][0] + o))

            if is_halanta_offset(o):
                itrans = ""
                if len(itrans_text) > 0:
                    itrans_text[-1] = itrans_text[-1][:-1]
            elif is_vowel_sign_offset(o) and len(itrans_text) > 0:
                itrans_text[-1] = itrans_text[-1][:-1]

            itrans_text.append(itrans)

        itrans_text = [char for char in itrans_text if char != ""]

        return itrans_text
    else:
        return text
