import json
import os
import logging
from typing import List, Dict
import string
from nltk.stem.porter import PorterStemmer
import collections
import time
import math


def stop_word_list():  # Construct stop word list
    with open("stop_words.txt", "r") as stop_word_file:
        return stop_word_file.read().splitlines()


def tokenizer(topic_info_dict: Dict[str, str]) -> Dict[str, List[str]]:
    tokens_dict: Dict[str, List[str]] = {}  # {paper_id: token_list}
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
    parent_input_path = "/Users/apple/Desktop/input_data/document_parses/pdf_json"  # write path to pdf_jsons folder
    pdf_jsons: List[str] = os.listdir(parent_input_path)
    # log_f = open(os.path.join(os.path.join(os.path.dirname(__file__), "log"), "output_log.txt"), "wb")
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
        # log_f.write("Parsed file: {0}\n".format(file_name).encode("utf-8"))
    # log_f.close()
    return topic_info_dict


def calculate_idf(df_dict: Dict[str, int], num_of_docs: int) -> Dict[str, float]:
    idf_dict: Dict[str, float] = {}
    for word in df_dict:
        idf_dict[word] = math.log(num_of_docs / df_dict[word])
    return idf_dict


def calculate_df(tokens_dict: Dict[str, List[str]]) -> Dict[str, int]:
    df_dict: Dict[str, int] = {}  # {token: doc. freq.}
    for doc_id in tokens_dict:
        words_set = list(set(tokens_dict[doc_id]))
        for word in words_set:
            if word not in df_dict:
                df_dict[word] = 1
            else:
                df_dict[word] += 1
    return df_dict


def calculate_tf_weight(tokens_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    tf_dict = {}  # {doc_id: {token: frequency}
    for doc_id in tokens_dict:
        tf_dict[doc_id] = dict(collections.Counter(tokens_dict[doc_id]))
    return tf_dict


STOP_WORDS = stop_word_list()

if __name__ == "__main__":
    begin_time = time.time()
    topic_info_dict: Dict[str, str] = extract_files()
    before_tf = time.time() - begin_time
    print("File extraction is ended. Time passed: {0}".format(before_tf))

    tokenization_time = time.time()
    tokens_dict, all_tokens = tokenizer(topic_info_dict)  # Dict[str, List[str]], List[str]
    tokenization_time = time.time() - tokenization_time
    print("Tokenization is ended. Time passed: {0}".format(tokenization_time))

    before_tf = time.time()
    tf_dict: Dict[str, Dict[str, int]] = calculate_tf_weight(tokens_dict)
    tf_time = time.time() - before_tf
    print("Calculating TF is ended. Time passed: {0}".format(tf_time))

    before_df = time.time()
    df_dict: Dict[str, int] = calculate_df(tokens_dict)
    df_time = time.time() - before_df
    print("Calculating DF is ended. Time passed: {0}".format(df_time))

    before_idf = time.time()
    idf_dict: Dict[str, float] = calculate_idf(df_dict, len(tokens_dict))
    idf_time = time.time() - before_idf
    print("Calculating IDF is ended. Time passed: {0}".format(idf_time))
