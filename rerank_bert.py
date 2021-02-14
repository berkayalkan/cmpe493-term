import math
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict, List

import nltk
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
import pickle

import file_operation
import tokenizer

stop_words = ["am", "is", "are", "the", "a", "this", "that", "e"]


def calculate_similarity(vectorizer, sentences, num_of_sentence_query, num_of_sentence_doc):
    try:
        vectorizer.bert(sentences)
        vectors_bert = vectorizer.vectors
        dist = 0
        for i in range(num_of_sentence_query):
            for j in range(num_of_sentence_query, num_of_sentence_query + num_of_sentence_doc):
                dist += spatial.distance.cosine(vectors_bert[i], vectors_bert[j])
        dist = dist / (num_of_sentence_query * num_of_sentence_doc)
        # d[t_id][d_id] = dist
        return dist
    except:
        try:
            new_sentences = []
            for sentence in sentences:
                word_list = sentence.split()
                result_words = [word for word in word_list if word.lower() not in stop_words]
                new_sentence = " ".join(result_words)
                new_sentences.append(new_sentence)
            vectorizer.bert(new_sentences)
            vectors_bert = vectorizer.vectors
            dist = 0
            for i in range(num_of_sentence_query):
                for j in range(num_of_sentence_query, num_of_sentence_query + num_of_sentence_doc):
                    dist += spatial.distance.cosine(vectors_bert[i], vectors_bert[j])
            dist = dist / (num_of_sentence_query * num_of_sentence_doc)
            return dist
        except:
            try:
                new_sentences = []
                new_num_of_sentence_query = num_of_sentence_query
                new_num_of_sentence_doc = num_of_sentence_doc
                i = 0
                for sentence in sentences:
                    two_point_list = sentence.split(':')
                    new_sentences += two_point_list
                    if i < num_of_sentence_query:
                        new_num_of_sentence_query += (len(two_point_list) - 1)
                    else:
                        new_num_of_sentence_doc += (len(two_point_list) - 1)
                    i += 1
                vectorizer.bert(new_sentences)
                vectors_bert = vectorizer.vectors
                dist = 0
                for i in range(new_num_of_sentence_query):
                    for j in range(new_num_of_sentence_query, new_num_of_sentence_query + new_num_of_sentence_doc):
                        dist += spatial.distance.cosine(vectors_bert[i], vectors_bert[j])
                dist = dist / (num_of_sentence_query * num_of_sentence_doc)
                return dist
            except:
                new_sentences = []
                new_num_of_sentence_query = num_of_sentence_query
                new_num_of_sentence_doc = num_of_sentence_doc
                i = 0
                for sentence in sentences:
                    words = sentence.split()
                    how_many = math.floor(len(words) / 100)
                    new_sentence = ""
                    for j in range(len(words)):
                        new_sentence = new_sentence + words[j] + ' '
                        if (j % 99 == 0 or j == len(words) - 1) and j != 0:
                            new_sentence = new_sentence[:-1] + '.'
                            new_sentences.append(new_sentence)
                            new_sentence = ""
                    if i < num_of_sentence_query:
                        new_num_of_sentence_query += how_many
                    else:
                        new_num_of_sentence_doc += how_many
                    i += 1
                vectorizer.bert(new_sentences)
                vectors_bert = vectorizer.vectors
                dist = 0
                for i in range(new_num_of_sentence_query):
                    for j in range(new_num_of_sentence_query, new_num_of_sentence_query + new_num_of_sentence_doc):
                        dist += spatial.distance.cosine(vectors_bert[i], vectors_bert[j])
                dist = dist / (num_of_sentence_query * num_of_sentence_doc)
                return dist


if __name__ == "__main__":
    f = open("input/bm25_all.pickle", "rb")
    bm25_dict = pickle.load(f)
    f.close()

    topic_info_dict: Dict[str, str] = file_operation.extract_file()

    result_dict = {}

    tokens_dict: Dict[str, List[str]] = tokenizer.tokenize_bert(topic_info_dict)

    train_query, test_query = file_operation.extract_queries_for_bert()

    nltk.download('punkt')
    for topic_id in train_query:
        result_dict[topic_id] = {}
        train_query[topic_id] = nltk.tokenize.sent_tokenize(train_query[topic_id])

    vectorizer = Vectorizer()
    for topic_id in train_query:
        topic_id = str(topic_id)
        executor = ThreadPoolExecutor(len(tokens_dict))
        num_of_sentence_query = len(train_query[topic_id])
        i = 0
        ths = []

        temp_bm25_dict = bm25_dict[topic_id]
        sorted_bm25 = list(dict(sorted(temp_bm25_dict.items(), key=lambda item: item[1], reverse=True)).keys())[:100]

        for doc_id in sorted_bm25:
            num_of_sentence_doc = len(tokens_dict[doc_id])
            all_sentences = train_query[topic_id] + tokens_dict[doc_id]

            sim = calculate_similarity(vectorizer, all_sentences, num_of_sentence_query, num_of_sentence_doc)
            result_dict[topic_id][doc_id] = sim

    f = open('input/bert.pickle', 'wb')
    pickle.dump(result_dict, f)
    f.close()
