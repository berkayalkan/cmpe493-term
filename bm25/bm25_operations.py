import pickle
import time
import tokenizer
from datetime import datetime
import calculations
from typing import List, Dict
import main
import file_operation
import os

if __name__ == "__main__":

    begin_time = time.time()
    os.chdir(r'..\ ')
    topic_info_dict: Dict[str, str] = file_operation.extract_file()
    before_tf = time.time() - begin_time
    print("File extraction is ended. Time passed: {0}".format(before_tf))
    tokenization_time = time.time()
    tokens_dict: Dict[str, List[str]] = tokenizer.tokenize(topic_info_dict)

    query_token_dict = main.extract_queries()[0]
    query_tokens: Dict[str, List[str]] = tokenizer.tokenize(query_token_dict)
    tokenization_time = time.time() - tokenization_time
    print("Tokenization is ended. Time passed: {0}".format(tokenization_time))
    start = datetime.now()
    print("start at: " + str(start))

    bm25_result_dict = {}
    bm25_first_dict = {}
    for topic in query_token_dict:
        bm25_first_dict[topic] = calculations.bm25_score(tokens_dict.values(), query_tokens[topic])
    for topic in bm25_first_dict:
        res_dict = {}
        i = 0
        for doc_id in tokens_dict:
            res_dict[doc_id] = bm25_first_dict[topic][i]
            i += 1
        bm25_result_dict[topic] = res_dict

    f = open('input/bm25_trained.pickle', 'wb')
    pickle.dump(bm25_result_dict, f)
    f.close()

    file_operation.write_results(bm25_result_dict)

