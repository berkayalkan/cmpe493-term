import multiprocessing
from typing import Dict, List
import gensim
import tqdm
import os


def calculate_doc2vec(tokens_dict: Dict[str, List[str]], train_token_dict: Dict[str, List[str]]) -> \
        Dict[str, Dict[str, float]]:
    corpus = list(create_train_corpus(tokens_dict))
    query_corpus = list(create_train_corpus(train_token_dict))
    if not os.path.exists("input/20_doc2vec_model"):
        cores = multiprocessing.cpu_count()
        model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=20, workers=cores)
        model.build_vocab([x for x in tqdm.tqdm(corpus)])
        print('\033[32m' + "Vocabulary is built." + '\033[0m')
        model.train([x for x in tqdm.tqdm(corpus)], total_examples=model.corpus_count, epochs=model.epochs)
        print('\033[32m' + "Model is trained." + '\033[0m')
        model.save('input/20_doc2vec_model')
        print('\033[32m' + "Model is saved." + '\033[0m')
    else:
        model = gensim.models.doc2vec.Doc2Vec.load('input/20_doc2vec_model')

    result_dict: Dict[str, Dict[str, float]] = {}
    print('\033[32m' + "Starting to calculate the cosine values." + '\033[0m')
    for i in tqdm.tqdm(range(len(query_corpus))):
        query_id = query_corpus[i].tags[0]
        result_dict[query_id] = {}
        inferred_vector = model.infer_vector(query_corpus[i].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        for doc_id, sim_value in sims:
            result_dict[query_id][doc_id] = sim_value

    print('\033[32m' + "Result dictionary is returned." + '\033[0m')
    return result_dict


def create_train_corpus(tokens_dict: Dict[str, List[str]]):
    for doc_id in tokens_dict:
        yield gensim.models.doc2vec.TaggedDocument(tokens_dict[doc_id], [doc_id])
