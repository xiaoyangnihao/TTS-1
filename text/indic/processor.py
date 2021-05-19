"""Frontend processor for Indic text"""

# Set of symbols
_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"
_punctuation = "^!\'(),-.:;? "
_roman_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Get list of symbols to be used for text processing
symbols = [_pad, _eos, _unk] + list(_punctuation) + list(_roman_characters)

# Map symbols to integer indices
symbol_to_id = {symb: index for index, symb in enumerate(symbols)}


def text_to_sequence(text):
    """Converts text to a sequence of IDs corresponding to the symbols in the text
    """
    text_seq = [
        symbol_to_id[char]
        if char in symbol_to_id else symbol_to_id["_UNK_"]
        for char in text
    ]

    return text_seq
