import pickle

if __name__=="__main__":
    f = open("input/official_results.txt", "r")
    lines = f.readlines()
    f.close()

    f = open("input/bm25_all.pickle", "rb")
    bm25_dict = pickle.load(f)
    f.close()

    dic = {}
    for topic_id in range(1, 51):
        temp_bm25_dict = bm25_dict[str(topic_id)]
        sorted_bm25 = list(dict(sorted(temp_bm25_dict.items(), key=lambda item: item[1], reverse=True)).keys())[:100]
        dic[str(topic_id)] = sorted_bm25

    for line in lines:
        with open("input/official_results_top100.txt", "a") as f:
            tokens = line.split()
            topic_id = tokens[0]
            if tokens[2] in dic[topic_id]:
                f.write(line)

    print()
