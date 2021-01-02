import collections
import logging
from typing import List, Dict
import string
from nltk.stem.porter import PorterStemmer
import time
import math
import pandas as pd


def stop_word_list() -> List[str]:  # Construct stop word list
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


def extract_file() -> Dict[str, str]:
    df: pd.DataFrame = pd.read_csv("input/metadata.csv", index_col=None, usecols=["cord_uid", "title", "abstract"])\
        .fillna("")
    df["text"] = df["title"] + " " + df["abstract"]
    topic_info_dict: Dict[str, str] = dict(pd.Series(df.text.values, index=df.cord_uid).to_dict())
    return topic_info_dict  # {doc_id: title_plus_abstract}


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


def calculate_tf_weight(tokens_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    tf_dict: Dict[str, Dict[str, float]] = {x: {} for x in tokens_dict}  # {doc_id: {token: tf}
    for doc_id in tokens_dict:
        doc_freq: Dict[str, int] = dict(collections.Counter(tokens_dict[doc_id]))  # {token: frequency}
        for token in doc_freq:
            tf_dict[doc_id][token] = 1.0 + math.log(doc_freq[token])
    return tf_dict


def calculate_score(tf_dict: Dict[str, Dict[str, float]], idf_dict: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    for doc_id in tf_dict:
        for token in tf_dict[doc_id]:
            idf_value: float = idf_dict[token]
            tf_value: float = tf_dict[doc_id][token]
            tf_dict[doc_id][
                token] = idf_value * tf_value  # tf_dict now keeps the score of the token instead of tf value
    return tf_dict


def calculate_normalization(score_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    total: float = 0
    for doc_id in score_dict:
        for token in score_dict[doc_id]:
            val: float = score_dict[doc_id][token]
            total += val**2

        total_sqrt = math.sqrt(total)

        for token in score_dict[doc_id]:
            score_dict[doc_id][token] /= total_sqrt
    return score_dict


def cos_calculator(doc_dict: Dict[str, float], query_dict: Dict[str, float]):  # one document vs. query
    val: float = 0
    for token in query_dict:
        if token in doc_dict:
            val += query_dict[token]*doc_dict[token]
    return val


STOP_WORDS: List[str] = stop_word_list()

if __name__ == "__main__":
    begin_time = time.time()
    topic_info_dict: Dict[str, str] = extract_file()
    before_tf = time.time() - begin_time
    print("File extraction is ended. Time passed: {0}".format(before_tf))

    tokenization_time = time.time()
    tokens_dict: Dict[str, List[str]] = tokenizer(topic_info_dict)  # Dict[str, List[str]], List[str]
    tokenization_time = time.time() - tokenization_time
    print("Tokenization is ended. Time passed: {0}".format(tokenization_time))

    before_tf = time.time()
    tf_dict: Dict[str, Dict[str, float]] = calculate_tf_weight(tokens_dict)
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

    before_score = time.time()
    score_dict: Dict[str, Dict[str, float]] = calculate_score(tf_dict, idf_dict)
    score_time = time.time() - before_score
    print("Calculating SCORE is ended. Time passed: {0}".format(score_time))

    # AFTER LENGTH NORMALIZATION
    before_normalization = time.time()
    normalized_dict:  Dict[str, Dict[str, float]] = calculate_normalization(score_dict)
    normalization_time = time.time() - before_normalization
    print("Calculating NORMALIZATION is ended. Time passed: {0}".format(normalization_time))
