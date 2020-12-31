import string
from typing import List, Dict, TypeVar
from stemmer import PorterStemmer
T = TypeVar('T')


def stop_word_list():  # Construct stop word list
    with open("stopwords.txt", "r") as f:
        lines = f.readlines()
        words = []
        for line in lines:
            words.append(line.strip())
        return words


def tokenizer(topic_info_dict: Dict[str, T]):
    punctuation_list = list(string.punctuation)
    stemmer = PorterStemmer()
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
        info_words = [stemmer.stem(word) for word in info_words if word not in STOP_WORDS and
                      (word.isnumeric() or len(word) > 1)]

        topic_info_dict[key] = info_words
    return topic_info_dict


STOP_WORDS = stop_word_list()


if __name__ == "__main__":
    tokens_dict: Dict[str, List[str]] = tokenizer({})
