import math
import collections
import time
from typing import List, Dict
from rank_bm25 import BM25Okapi


def calculate_idf(df_dict: Dict[str, int], num_of_docs: int) -> Dict[str, float]:
    """Calculates idf"""
    idf_dict: Dict[str, float] = {}
    for word in df_dict:
        idf_dict[word] = math.log(num_of_docs / df_dict[word])
    return idf_dict


def calculate_df(tokens_dict: Dict[str, List[str]]) -> Dict[str, int]:
    """Calculates df"""
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
    """Calculates tf weight"""
    tf_dict: Dict[str, Dict[str, float]] = {x: {} for x in tokens_dict}  # {doc_id: {token: tf}
    for doc_id in tokens_dict:
        doc_freq: Dict[str, int] = dict(collections.Counter(tokens_dict[doc_id]))  # {token: frequency}
        for token in doc_freq:
            tf_dict[doc_id][token] = 1.0 + math.log(doc_freq[token])
    return tf_dict


def calculate_score(tf_dict: Dict[str, Dict[str, float]], idf_dict: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Calculates tf*idf score"""
    for doc_id in tf_dict:
        for token in tf_dict[doc_id]:
            idf_value: float = idf_dict[token]
            tf_value: float = tf_dict[doc_id][token]
            tf_dict[doc_id][
                token] = idf_value * tf_value  # tf_dict now keeps the score of the token instead of tf value
    return tf_dict


def calculate_normalization(score_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Normalization step"""
    total: float = 0
    for doc_id in score_dict:
        for token in score_dict[doc_id]:
            val: float = score_dict[doc_id][token]
            total += val ** 2

        total_sqrt = math.sqrt(total)

        for token in score_dict[doc_id]:
            score_dict[doc_id][token] /= total_sqrt
        total = 0
    return score_dict


def cos_calculator(doc_dict: Dict[str, float], query_dict: Dict[str, float]):  # one document vs. query
    """Cosine simularity function"""
    val: float = 0
    for token in query_dict:
        if token in doc_dict:
            val += query_dict[token] * doc_dict[token]
    return val


def bm25_scores(tokenized_corpus_dict, topics_query):
    bm25_scores_dict = {}
    for topic in topics_query:
        for doc in tokenized_corpus_dict:
            bm25_scores_dict[doc] = bm25_score(tokenized_corpus_dict[doc], topics_query[topic])
    return bm25_scores_dict


def bm25_score(tokenized_corpus, tokenized_query):
    bm25 = BM25Okapi(tokenized_corpus)
    score = bm25.get_scores(tokenized_query)
    return score


def compare(normalized_dict: Dict[str, Dict[str, float]], train_normalized_dict: Dict[str, Dict[str, float]]) \
        -> Dict[str, Dict[str, float]]:
    result_dict: Dict[str, Dict[str, float]] = {}
    for topic_id in train_normalized_dict:
        for doc_id in normalized_dict:
            if topic_id not in result_dict:
                result_dict.update({topic_id: {}})
            result_dict[topic_id].update({doc_id: cos_calculator(normalized_dict[doc_id],
                                                                 train_normalized_dict[topic_id])})
    return result_dict


def run_tfidf(tokens_dict: Dict[str, List[str]], train_token_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    tf_dict: Dict[str, Dict[str, float]] = calculate_tf_weight(tokens_dict)
    df_dict: Dict[str, int] = calculate_df(tokens_dict)
    idf_dict: Dict[str, float] = calculate_idf(df_dict, len(tokens_dict))
    score_dict: Dict[str, Dict[str, float]] = calculate_score(tf_dict, idf_dict)
    # AFTER LENGTH NORMALIZATION
    normalized_dict: Dict[str, Dict[str, float]] = calculate_normalization(score_dict)

    train_tf_dict: Dict[str, Dict[str, float]] = calculate_tf_weight(train_token_dict)
    train_df_dict: Dict[str, int] = calculate_df(train_token_dict)
    train_idf_dict: Dict[str, float] = calculate_idf(train_df_dict, len(train_token_dict))
    train_score_dict: Dict[str, Dict[str, float]] = calculate_score(train_tf_dict, train_idf_dict)
    train_normalized_dict: Dict[str, Dict[str, float]] = calculate_normalization(train_score_dict)

    before_result = time.time()
    result_dict: Dict[str, Dict[str, float]] = compare(normalized_dict, train_normalized_dict)
    result_time = time.time() - before_result
    print("Calculating RESULT is ended. Time passed: {0}".format(result_time))
    return result_dict
