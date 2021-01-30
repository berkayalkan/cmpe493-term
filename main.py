import time
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
        train_query[str(index + 1)] = query_text  # şimdilik sadece bu
        """if index % 2 == 0:
            train_query[str(index + 1)] = query_text
        else:
            test_query[str(index + 1)] = query_text"""
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


if __name__ == "__main__":
    begin_time = time.time()
    topic_info_dict: Dict[str, str] = file_operation.extract_file()
    before_tf = time.time() - begin_time
    print("File extraction is ended. Time passed: {0}".format(before_tf))

    tokenization_time = time.time()
    tokens_dict: Dict[str, List[str]] = tokenizer.tokenize(topic_info_dict)  # Dict[str, List[str]], List[str]
    tokenization_time = time.time() - tokenization_time
    print("Tokenization is ended. Time passed: {0}".format(tokenization_time))

    """f = open('input/doc_tokens.pickle', 'rb')
    tokens_dict = pickle.load(f)
    f.close()"""

    tf_dict: Dict[str, Dict[str, float]] = calculations.calculate_tf_weight(tokens_dict)
    df_dict: Dict[str, int] = calculations.calculate_df(tokens_dict)
    idf_dict: Dict[str, float] = calculations.calculate_idf(df_dict, len(tokens_dict))
    score_dict: Dict[str, Dict[str, float]] = calculations.calculate_score(tf_dict, idf_dict)
    # AFTER LENGTH NORMALIZATION
    normalized_dict: Dict[str, Dict[str, float]] = calculations.calculate_normalization(score_dict)

    train_query, test_query = extract_queries()
    train_token_dict: Dict[str, List[str]] = tokenizer.tokenize(train_query)

    """f = open('input/topic_tokens.pickle', 'rb')
    train_token_dict = pickle.load(f)
    f.close()"""

    train_tf_dict: Dict[str, Dict[str, float]] = calculations.calculate_tf_weight(train_token_dict)
    train_df_dict: Dict[str, int] = calculations.calculate_df(train_token_dict)
    train_idf_dict: Dict[str, float] = calculations.calculate_idf(train_df_dict, len(train_token_dict))
    train_score_dict: Dict[str, Dict[str, float]] = calculations.calculate_score(train_tf_dict, train_idf_dict)
    train_normalized_dict: Dict[str, Dict[str, float]] = calculations.calculate_normalization(train_score_dict)

    before_result = time.time()
    result_dict: Dict[str, Dict[str, float]] = compare(normalized_dict, train_normalized_dict)
    result_time = time.time() - before_result
    print("Calculating RESULT is ended. Time passed: {0}".format(result_time))

    before_output = time.time()
    file_operation.write_results(result_dict)
    output_time = time.time() - before_output
    print("Calculating OUTPUT is ended. Time passed: {0}".format(output_time))
