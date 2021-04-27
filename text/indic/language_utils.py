"""Language utilities"""

import re

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


def load_phone_set(language_code):
    """Load the appropriate phone set (based on the language)
    """
    # codepoint_phone_mapping = {}

    # Read the phoneset file and get the data
    language_string = language_codes[language_code]

    with open(f"CPS_Mapping/{language_string}" + ".txt") as file_reader:
        phone_set_data = file_reader.readlines()

    phone_set_data = [line.strip("\n").rstrip("\t") for line in phone_set_data]
    phone_set_data = [re.split(r"\t", line) for line in phone_set_data]

    codepoint_phone_mapping = {
        int(codepoint, 16): phone
        for codepoint, phone, _ in phone_set_data
    }

    return codepoint_phone_mapping
