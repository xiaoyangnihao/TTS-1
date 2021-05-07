"""Frontend processor for Indic text"""

from text.indic.language_utils import (offset_itrans_mapping,
                                       unicode_to_itrans_tranliteration)

# Set of symbols
_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"
_punctuation = "!\'(),-.:;? "
_itrans_symbols = offset_itrans_mapping.values()

symbols = [_pad, _eos, _unk] + list(_punctuation) + list(_itrans_symbols)

# Map symbols to integer indices
symbol_to_id_indic = {symb: index for index, symb in enumerate(symbols)}


def text_to_sequence(text, lang_code):
    """Converts text to a sequence of IDs corresponding to the symbols in the text
    """
    text = unicode_to_itrans_tranliteration(text, lang_code)
    text_seq = [
        symbol_to_id_indic[char]
        if char in symbol_to_id_indic else symbol_to_id_indic["_UNK_"]
        for char in text
    ]

    return text_seq
