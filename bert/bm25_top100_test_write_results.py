import pickle
from file_operation import write_results

if __name__=="__main__":
    f = open("input/bm25_all.pickle", "rb")
    bm25_dict = pickle.load(f)
    f.close()

    bm25_top100_test_dict = {}
    for i in range(2, 51, 2):
        temp_bm25_dict = bm25_dict[str(i)]
        sorted_bm25 = list(dict(sorted(temp_bm25_dict.items(), key=lambda item: item[1], reverse=True)).keys())[:100]
        temp_dict = {}
        for doc_id in sorted_bm25:
            temp_dict[doc_id] = bm25_dict[str(i)][doc_id]
        bm25_top100_test_dict[str(i)] = temp_dict

    write_results(bm25_top100_test_dict)
