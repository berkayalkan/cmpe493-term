import json
import os
import logging
from typing import List, Dict
import string
from stemmer import PorterStemmer


def stop_word_list():  # Construct stop word list
    with open("stopwords.txt", "r") as f:
        lines = f.readlines()
        words = []
        for line in lines:
            words.append(line.strip())
        return words


def tokenizer(topic_info_dict: Dict[str, str]) -> Dict[str, List[str]]:
    tokens_dict: Dict[str, List[str]] = None
    punctuation_list = list(string.punctuation)
    stemmer = PorterStemmer()
    for key in topic_info_dict:
        info = topic_info_dict[key]
        info = info.lower()
        # Replace punctuations with space character in the document information
        for punctuation in punctuation_list:
            if punctuation == ".":
                info = info.replace(punctuation, "")
                continue
            info = info.replace(punctuation, " ")

        # Take terms as list
        info_words = info.split()

        # Remove stop words from the document information
        # If word's length is 1 and word is not an one digit number, remove the word from list
        info_words = [stemmer.stem(word) for word in info_words if word not in STOP_WORDS and
                      (word.isnumeric() or len(word) > 1)]

        tokens_dict[key] = info_words
    return tokens_dict


def extract_files() -> Dict[str, str]:
    topic_info_dict: Dict[str, str] = {}  # {doc_id: abstract_plus_title}
    # write your path to pdf_jsons folder
    parent_input_path = "/Users/apple/Desktop/input_data/document_parses/pdf_json"
    pdf_jsons: List[str] = os.listdir(parent_input_path)
    log_f = open(os.path.join(os.path.join(os.path.dirname(__file__), "log"), "output_log.txt"), "wb")
    for file_name in pdf_jsons:
        file_path = os.path.join(parent_input_path, file_name)
        with open(file_path) as json_f:
            try:
                file_dict = json.load(json_f)
            except json.decoder.JSONDecodeError as ex:
                logging.error("Error happened while parsing {0}: {1}".format(file_name, ex))
                continue
            extracted_info: str = file_dict["metadata"]["title"]
            for abstract_texts in file_dict["abstract"]:
                extracted_info += " " + abstract_texts["text"]
            paper_id: str = file_dict["paper_id"]
            if extracted_info is not None and extracted_info.strip() != "" and paper_id is not None \
                    and paper_id.strip() != "":  # no time for useless papers :)
                topic_info_dict[file_dict["paper_id"]] = extracted_info
        log_f.write("Parsed file: {0}\n".format(file_name).encode("utf-8"))
    log_f.close()
    return topic_info_dict


STOP_WORDS = stop_word_list()


if __name__ == "__main__":
    topic_info_dict: Dict[str, str] = extract_files()
    tokens_dict: Dict[str, List[str]] = tokenizer(topic_info_dict)
