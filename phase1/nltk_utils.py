import nltk
import numpy as np
#uncomment below for "nltk.word_tokenize"
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

#--------------------text preprocessing--------------------
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

stemmer = PorterStemmer()

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    '''
    eg.
    sentence = ["hello","how","are","you"]
    words = ["hi","hello","I","you","bye","thank","cool"]
    bow   = [  0     1     0    1     0      0       0]
    '''
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bow = np.zeros(len(all_words), dtype = np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bow[idx] = 1.0

    return bow


# def main():
#     print("Begin to chat!")
    
# if __name__ == "__main__":
#     main()