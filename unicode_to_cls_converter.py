"""Unicode to Common Label Set converter"""

import argparse
import re

lang_code_mapping = {
    "as": "Assamese",
    "be": "Bengali",
    "gj": "Gujarati",
    "hi": "Hindi",
    "ma": "Marathi",
    "rj": "Rajasthani",
    "ta": "Tamil",
    "te": "Telugu"
}


def load_unicode_cls_mapping(lang_code):
    """Load the unicode to cls mapping for a particular language
    """
    if lang_code in lang_code_mapping:
        language = lang_code_mapping[lang_code]
        with open(f"text/indic/Unicode_CLS_Mapping/{language}.txt",
                  "r") as file_reader:
            mapping = file_reader.readlines()

        mapping = [line.strip("\n") for line in mapping]

        for line in mapping:
            codepoint, phone, phone_type = line.split()
            print(codepoint, phone, phone_type)
    else:
        raise Exception(f"Language code {lang_code} not supported")


def convert_prompts_file(in_file, out_file, lang_code):
    """Convert the prompts file from unicode to common label set
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
