import nltk
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
    pass







# def main():
#     print("Begin to chat!")
    
# if __name__ == "__main__":
#     main()