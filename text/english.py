"""Frontend processor for English text"""

from text.english_normalization import (add_punctuation, collapse_whitespace,
                                        expand_abbreviations,
                                        normalize_numbers)

# Set of symbols
_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"
_punctuation = "!\'(),-.:;? "
_english_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Get list of symbols to be used for text processing
symbols = [_pad, _eos, _unk] + list(_punctuation) + list(_english_characters)

# Map symbols to integer indices
symbol_to_id = {symb: index for index, symb in enumerate(symbols)}
id_to_symbol = {index: symb for index, symb in enumerate(symbols)}


def parse_text(text):
    """Parse the text and perform text normalization
    """
    text = add_punctuation(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)

    return text


def text_to_sequence(text):
    """Converts text to a sequence of IDs corresponding to the symbols in the text
    """
    text = parse_text(text)

    text_seq = [
        symbol_to_id[char] if char in symbol_to_id else symbol_to_id["_UNK_"]
        for char in text
    ]

    return text_seq
