import string
import file_operation
from typing import List, Dict
from nltk.stem.porter import PorterStemmer
import nltk.data


def tokenize(topic_info_dict: Dict[str, str]) -> Dict[str, List[str]]:
    STOP_WORDS: List[str] = file_operation.stop_word_list()

    tokens_dict: Dict[str, List[str]] = {}  # {paper_id: token_list}
    punctuation_list = list(string.punctuation)
    stemmer = PorterStemmer(mode="MARTIN_EXTENSIONS")
    for key in topic_info_dict:
        info = topic_info_dict[key]
        info = info.lower()
        # Replace punctuations with space character in the document information
        for punctuation in punctuation_list:
            if punctuation == ".":
                info = info.replace(punctuation, "")
                continue
            info = info.replace(punctuation, " ")

        # Take terms as list
        info_words = info.split()

        # Remove stop words from the document information
        # If word's length is 1 and word is not an one digit number, remove the word from list
        info_words = [stemmer.stem(word) for word in info_words
                      if word not in STOP_WORDS and (word.isnumeric() or len(word) > 1)]

        tokens_dict[key] = info_words
    return tokens_dict


def tokenize_bert(topic_info_dict: Dict[str, str]) -> Dict[str, List[str]]:
    tokens_dict: Dict[str, List[str]] = {}
    nltk.download('punkt')
    for topic in topic_info_dict:
        tokens_dict[topic] = nltk.tokenize.sent_tokenize(topic_info_dict[topic])
    return tokens_dict
