import multiprocessing
from typing import Dict, List
import gensim
import collections
import tqdm
from sklearn import utils
import os
import json


def calculate_doc2vec(tokens_dict: Dict[str, List[str]], train_token_dict: Dict[str, List[str]]) -> \
        Dict[str, Dict[str, float]]:
    corpus = list(create_train_corpus(tokens_dict))
    query_corpus = list(create_train_corpus(train_token_dict))
    if not os.path.exists("input/18_doc2vec_model"):
        cores = multiprocessing.cpu_count()
        model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=18, alpha=0.065, min_alpha=0.065, workers=cores)
        model.build_vocab([x for x in tqdm.tqdm(corpus)])
        # model.train([x for x in tqdm.tqdm(corpus)], total_examples=model.corpus_count, epochs=model.epochs)

        print('\033[32m' + "Vocabulary is built." + '\033[0m')
        for epoch in range(30):
            model.train(utils.shuffle([x for x in tqdm.tqdm(corpus)]), total_examples=len(corpus), epochs=1)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        print('\033[32m' + "Model is trained." + '\033[0m')

        model.save('input/18_doc2vec_model')
        print('\033[32m' + "Model is saved." + '\033[0m')

    else:
        model = gensim.models.doc2vec.Doc2Vec.load('input/18_doc2vec_model')

    """# ASSESS
    ranks = []
    second_ranks = []
    for doc_id in tqdm.tqdm(range(len(corpus))):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(corpus[doc_id].tags[0])
        ranks.append(rank)

        second_ranks.append(sims[1])
    counter = collections.Counter(ranks)
    with open("output/assesment_doc2vec.json") as f:
        json.dump(counter, f)"""

    result_dict: Dict[str, Dict[str, float]] = {}
    print('\033[32m' + "Starting to calculate the cosine values." + '\033[0m')
    for i in tqdm.tqdm(range(len(query_corpus))):
        query_id = query_corpus[i].tags[0]
        result_dict[query_id] = {}
        inferred_vector = model.infer_vector(query_corpus[i].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        for doc_id, sim_value in sims:
            result_dict[query_id][doc_id] = sim_value

    print('\033[32m' + "Result Dict is returned" + '\033[0m')
    return result_dict


def create_train_corpus(tokens_dict: Dict[str, List[str]]):
    for doc_id in tokens_dict:
        yield gensim.models.doc2vec.TaggedDocument(tokens_dict[doc_id], [doc_id])
