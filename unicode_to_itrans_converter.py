"""Unicode to ITRANS converter"""

import argparse

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
    0x7d: "ॽ",
    0x7f: "a",
}

script_ranges = {
    "gu": [0x0a80, 0x0aff],
    "ta": [0x0b80, 0x0bff],
    "te": [0x0c00, 0x0c7f],
    "ml": [0x0d00, 0x0d7f],
    "hi": [0x0900, 0x097f],
    "mr": [0x0900, 0x097f],
    "rj": [0x0900, 0x097f],
    "bn": [0x0980, 0x09ff],
    "as": [0x0980, 0x09ff],
}

HALANTA_OFFSET = 0x4d


def is_vowel_sign_offset(c_offset):
    """Is the offset a vowel sign (maatraa)
    """
    return (c_offset >= 0x3e and c_offset <= 0x4c)


def is_halanta_offset(c_offset):
    """Is the offset a halanta
    """
    return (c_offset == HALANTA_OFFSET)


def char_to_offset(char, lang_code):
    """Get the offset corresponding to a character in the language (represented by lang_code)
    """
    return ord(char) - script_ranges[lang_code][0]


def unicode_to_itrans(unicode_text, lang_code):
    """Convert text in unicode to text in itrans
    """
    if lang_code in script_ranges:
        offsets = [char_to_offset(char, lang_code) for char in unicode_text]

        itrans_text = []
        for o in offsets:
            itrans = offset_itrans_mapping.get(
                o, chr(script_ranges[lang_code][0] + o))

            if is_halanta_offset(o):
                itrans = ""
                if len(itrans_text) > 0:
                    itrans_text.pop()
            elif is_vowel_sign_offset(o) and len(itrans_text) > 0:
                itrans_text.pop()

            itrans_text.extend(itrans)

        return "".join(itrans_text)
    else:
        return unicode_text


def convert_prompts_file(in_file, out_file, lang_code):
    """Convert the prompts file from unicode to itrans
    """
    # Read the unicode prompts file
    with open(in_file, "r", encoding="utf-8") as file_reader:
        unicode_prompts = file_reader.readlines()

    unicode_prompts = [line.strip("\n") for line in unicode_prompts]
    unicode_prompts = [line.split("|") for line in unicode_prompts]

    itrans_prompts = [(prompt_id, unicode_to_itrans(text, lang_code=lang_code))
                      for prompt_id, text in unicode_prompts]

    # Write the itrans converted prompts to output file
    file_writer = open(out_file, "w", encoding="utf-8")
    for prompt_id, text in itrans_prompts:
        line = prompt_id + "|" + text + "\n"
        file_writer.write(line)
    file_writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Unicode to ITRANS converter")

    parser.add_argument(
        "--unicode_prompts_file",
        help="Prompts file containing unicode text (to be converted to itrans)",
        required=True)

    parser.add_argument(
        "--itrans_prompts_file",
        help="Output file containing itrans text (converted from unicode)",
        required=True)

    parser.add_argument(
        "--lang_code",
        help="Code indicating the Indic language (gu, ta, te, hi, bn, as)",
        required=True)

    args = parser.parse_args()

    in_file = args.unicode_prompts_file
    out_file = args.itrans_prompts_file
    lang_code = args.lang_code

    convert_prompts_file(in_file, out_file, lang_code)
