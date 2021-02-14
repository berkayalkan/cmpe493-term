import pandas as pd
from typing import List, Dict
from bs4 import BeautifulSoup
from requests import get


def stop_word_list() -> List[str]:  # Construct stop word list
    with open("input/stop_words.txt", "r") as stop_word_file:
        return stop_word_file.read().splitlines()


def extract_file() -> Dict[str, str]:
    df_official: pd.DataFrame = pd.read_csv('input/official_results.txt', sep=" ", header=None)
    desired_cord_uids = list(set(df_official[2].values.tolist()))

    df: pd.DataFrame = pd.read_csv("input/metadata.csv", index_col=None, usecols=["cord_uid", "title", "abstract"]) \
        .fillna("")
    df = df[df.cord_uid.isin(desired_cord_uids)]
    df["text"] = df["title"] + " " + df["abstract"]
    # cord_id leri ayır tamam bu
    # sonra threshold belirle train için (dene), sonra relevant olanları, file a yazdırıp, trec_eval çalıştır.
    topic_info_dict: Dict[str, str] = dict(pd.Series(df.text.values, index=df.cord_uid).to_dict())

    return topic_info_dict  # {doc_id: title_plus_abstract}


def write_results(result_dict: Dict[str, Dict[str, float]]):
    """THRESHOLD = 0.2
    query-id Q0 document-id rank score STANDARD
    counter = 5
    while counter <= 5:
        line = b"""""
    with open("output/{0}_output.txt".format("test_bert_top100"), "wb") as out_file:

        for query_id in result_dict:
            for doc_id in result_dict[query_id]:
                out_file.write("{0} Q0 {1} 0 {2} STANDARD\n".format(query_id, doc_id, result_dict[query_id][doc_id])
                               .encode("utf-8"))

    """print("THRESHOLD IS ---> {0}".format(THRESHOLD))
    counter += 1
    THRESHOLD = counter/100"""
    print("write_results is ended.")


def extract_queries_for_bert() -> (Dict[str, str], Dict[str, str]):
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
