import os
import pickle
import time
from doc2vec import doc2vec
import tokenizer
import calculations
import file_operation
from requests import get
from typing import List, Dict
from bs4 import BeautifulSoup


def extract_queries() -> (Dict[str, str], Dict[str, str]):
    url = 'https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml'
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    train_query: Dict[str, str] = {}
    test_query: Dict[str, str] = {}
    query_containers = html_soup.find_all('query')  # type = bs4.element.ResultSet
    question_containers = html_soup.find_all('question')
    narrative_containers = html_soup.find_all('narrative')

    for index in range(len(query_containers)):
        query_text: str = query_containers[index].text + " " + question_containers[index].text + " " \
                          + narrative_containers[index].text
        if index % 2 == 0:
            train_query[str(index + 1)] = query_text
        else:
            test_query[str(index + 1)] = query_text
    return train_query, test_query


def compare(normalized_dict: Dict[str, Dict[str, float]], train_normalized_dict: Dict[str, Dict[str, float]]) \
        -> Dict[str, Dict[str, float]]:
    result_dict: Dict[str, Dict[str, float]] = {}
    for topic_id in train_normalized_dict:
        for doc_id in normalized_dict:
            if topic_id not in result_dict:
                result_dict.update({topic_id: {}})
            result_dict[topic_id].update({doc_id: calculations.cos_calculator(normalized_dict[doc_id],
                                                                              train_normalized_dict[topic_id])})
    return result_dict


def create_dictionaries() -> (Dict[str, List[str]], Dict[str, List[str]]):
    """Creates the train / test query dictionary and token dictionary of the metadata.csv"""
    if os.path.exists('input/doc_tokens.pickle'):
        f = open('input/doc_tokens.pickle', 'rb')
        tokens_dict = pickle.load(f)
        f.close()
    else:
        begin_time = time.time()
        topic_info_dict: Dict[str, str] = file_operation.extract_file()
        before_tf = time.time() - begin_time
        print("File extraction is ended. Time passed: {0}".format(before_tf))

        tokenization_time = time.time()
        tokens_dict: Dict[str, List[str]] = tokenizer.tokenize(topic_info_dict)  # Dict[str, List[str]], List[str]
        tokenization_time = time.time() - tokenization_time
        print("Tokenization is ended. Time passed: {0}".format(tokenization_time))

        print('\033[32m' + "Output file for tokens dictionary is creating under input directory." + '\033[0m')
        f = open('input/doc_tokens.pickle', 'wb')
        pickle.dump(tokens_dict, f)
        f.close()

    if os.path.exists("input/topic_tokens.pickle"):
        f = open('input/topic_tokens.pickle', 'rb')
        train_token_dict = pickle.load(f)  # RN the pickle is train
        f.close()
    else:
        train_query, test_query = extract_queries()
        # IMPORTANT IF YOU WANT TO TEST THE MODEL, CALL 'tokenizer.tokenize(test_query)'.
        train_token_dict: Dict[str, List[str]] = tokenizer.tokenize(test_query)

        print('\033[32m' + "Output file for train query tokens is creating under input directory." + '\033[0m')
        f = open('input/topic_tokens.pickle', 'wb')
        pickle.dump(train_token_dict, f)
        f.close()
    return tokens_dict, train_token_dict


if __name__ == "__main__":

    tokens_dict, train_token_dict = create_dictionaries()
    result_dict = doc2vec.calculate_doc2vec(tokens_dict, train_token_dict)  # doc2vec method
    # result_dict = calculations.run_tfidf(tokens_dict, train_token_dict)

    before_output = time.time()
    # for NO THRESHOLD, call write_results | for doc2vec(threshold + scale) call write_results_w_scale
    file_operation.write_results_w_scale(result_dict)
    output_time = time.time() - before_output
    print("Calculating OUTPUT is ended. Time passed: {0}".format(output_time))
