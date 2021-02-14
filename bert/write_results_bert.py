import pickle

from file_operation import write_results

if __name__=="__main__":
    f = open("input/bert.pickle", "rb")
    d = pickle.load(f)
    f.close()

    test_bert_dict = {}
    for i in range(2, 51, 2):
        test_bert_dict[str(i)] = d[str(i)]
    write_results(test_bert_dict)

    print()
