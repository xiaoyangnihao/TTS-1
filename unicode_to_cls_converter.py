"""Converter to convert Indic unicode text to common label set"""

language_codes = {
    "as": "Assamese",
    "be": "Bengali",
    "gj": "Gujarati",
    "hi": "Hindi",
    "mn": "Manipuri",
    "ma": "Marathi",
    "rj": "Rajasthani",
    "ta": "Tamil",
    "te": "Telugu"
}


def load_cls_mapping(lang_code):
    """Load the Unicode code point to CLS mapping for the specified language
    """
    cls_mapping_file = f"text/indic/CLS_mapping/{language_codes[lang_code]}.txt"

    with open(cls_mapping_file, "r", encoding="utf-8") as file_reader:
        data = file_reader.readlines()

    data = [line.strip("\n") for line in data]
    data = [line.split() for line in data]

    codepoint_phone_mapping = {
        int(unicode_codepoint, base=16): phone
        for unicode_codepoint, phone, _ in data
    }

    phone_type_mapping = {phone: phone_type for _, phone, phone_type in data}

    return codepoint_phone_mapping, phone_type_mapping


def unicode_to_cls(unicode_text, codepoint_phone_mapping, phone_type_mapping):
    """Convert unicode indic text to common label set
    """
    unicode_text = [char for char in unicode_text]

    cls_text = [
        codepoint_phone_mapping[ord(char)]
        if ord(char) in codepoint_phone_mapping else char
        for char in unicode_text
    ]

    return cls_text
