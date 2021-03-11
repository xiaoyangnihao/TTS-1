"""Text processing for English"""

import re
from itertools import islice

# Set of symbols
_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"
_punctuation = "!,.? "
_english_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_cmudict_symbols = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1',
    'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0',
    'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0',
    'IH1', 'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W',
    'Y', 'Z', 'ZH'
]
_arpabet = ["@" + s for s in _cmudict_symbols]

# Get list of symbols to be used for text processing and create mapping
symbols = [_pad, _eos
           ] + list(_punctuation) + list(_english_characters) + _arpabet

symbol_to_id = {symb: index for index, symb in enumerate(symbols)}
id_to_symbol = {index: symb for index, symb in enumerate(symbols)}

# Regex for common abbreviations
abbreviations = [(re.compile(fr"\b{abbreviation}\.",
                             re.IGNORECASE), replacement.upper())
                 for abbreviation, replacement in [
                     ("mrs", "missis"),
                     ("mr", "mister"),
                     ("dr", "doctor"),
                     ("st", "saint"),
                     ("co", "company"),
                     ("jr", "junior"),
                     ("maj", "major"),
                     ("gen", "general"),
                     ("drs", "doctors"),
                     ("rev", "reverend"),
                     ("lt", "lieutenant"),
                     ("hon", "honorable"),
                     ("sgt", "sergeant"),
                     ("capt", "captain"),
                     ("esq", "esquire"),
                     ("ltd", "limited"),
                     ("col", "colonel"),
                     ("ft", "fort"),
                     ("etc", "etcetera"),
                 ]]

# Regex patterns for puncutations
parentheses_pattern = re.compile(
    r"(?<=[.,!?] )[\(\[]|[\)\]](?=[.,!?])|^[\(\[]|[\)\]]$")
dash_pattern = re.compile(r"(?<=[.,!?] )-- ")
alt_entry_pattern = re.compile(r"(?<=\w)\((\d)\)")
tokenizer_pattern = re.compile(r"[\w\{\}']+|[.,!?]")


def expand_abbreviations(text):
    """Expand abbreviations
    """
    for pattern, replacement in abbreviations:
        text = pattern.sub(replacement, text)

    return text


def format_alt_entry(text):
    return alt_entry_pattern.sub(r"{\1}", text)


def replace_symbols(text):
    # replace semi-colons and colons with commas
    text = text.replace(";", ",")
    text = text.replace(":", ",")

    # replace dashes with commas
    text = dash_pattern.sub("", text)
    text = text.replace(" --", ",")
    text = text.replace(" - ", ", ")

    # split hyphenated words
    text = text.replace("-", " ")

    # use {#} to indicate alternate pronunciations
    text = format_alt_entry(text)

    # replace parentheses with commas
    text = parentheses_pattern.sub("", text)
    text = text.replace(")", ",")
    text = text.replace(" (", ", ")
    text = text.replace("]", ",")
    text = text.replace(" [", ", ")

    return text


def clean_text(text):
    text = text
    text = expand_abbreviations(text)
    text = replace_symbols(text)

    return text


def tokenize_text(text):
    return tokenizer_pattern.findall(text)


def load_cmudict():
    """Load the CMU pronunciation dictionary
    """
    with open("text/cmudict-0.7b.txt", encoding="ISO-8859-1") as file_reader:
        cmudict = (line.strip().split("  ")
                   for line in islice(file_reader, 126, 133905))

        cmudict = {
            format_alt_entry(word): pronunciation
            for word, pronunciation in cmudict
        }

    return cmudict


def parse_text(text, cmudict):
    """Parse the text and get the sequence of phonemes for words in the CMU dict. For OOV words backoff to character
    sequence instead of phoneme sequence
    """
    text = tokenize_text(clean_text(text))
    text = [
        " ".join(["@" + s for s in cmudict[word.upper()].split(" ")])
        if word.upper() in cmudict else " ".join(char for char in word)
        for word in text
    ]

    text = [word.split(" ") for word in text]
    text = [char for word in text for char in word]
    text.append(_eos)

    return text


def text_to_id(text, cmudict):
    """Convert text to a sequence of symbol ids
    """
    symbols = parse_text(text, cmudict)

    return [
        symbol_to_id[s] if s in symbol_to_id else symbol_to_id[_unk]
        for s in symbols
    ]
