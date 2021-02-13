import pandas as pd
from typing import List, Dict


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
    with open("output/{0}_output.txt".format("doc2vec_3"), "wb") as out_file:

        for query_id in result_dict:
            for doc_id in result_dict[query_id]:
                out_file.write("{0} Q0 {1} 0 {2} STANDARD\n".format(query_id, doc_id, result_dict[query_id][doc_id])
                               .encode("utf-8"))

    """print("THRESHOLD IS ---> {0}".format(THRESHOLD))
    counter += 1
    THRESHOLD = counter/100"""
    print("write_results is ended.")


def write_results_w_threshold(result_dict: Dict[str, Dict[str, float]]):
    THRESHOLD = 0.1
    # query-id Q0 document-id rank score STANDARD
    counter = 1
    while counter <= 7:
        with open("output/{0}_output.txt".format(counter), "wb") as out_file:

            for query_id in result_dict:
                for doc_id in result_dict[query_id]:
                    if float(result_dict[query_id][doc_id]) > THRESHOLD:
                        out_file.write(
                            "{0} Q0 {1} 0 {2} STANDARD\n".format(query_id, doc_id, result_dict[query_id][doc_id])
                                .encode("utf-8"))

        print("THRESHOLD IS ---> {0}".format(THRESHOLD))
        counter += 1
        THRESHOLD = counter/10
        print("write_results is ended.")
