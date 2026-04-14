import re
import logging
import string
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from config.nltk_setup import download_nltk_resources
from src.preprocess_review import preprocess

logging.basicConfig(level=logging.INFO)

logging.info("Downloading nltk resources")
download_nltk_resources()
logging.info("Download complete..")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    tokens = word_tokenize(text)
    return tokens

def word2vec():
    logging.info("Procesing the data...")
    cleaned, tokens, filtered, lemmatized = preprocess()

    corpus = [' '.join(words) for words in lemmatized]
    processed_corpus= corpus*200

    processed_corpus = [preprocess_text(text) for text in processed_corpus]
    processed_corpus
    logging.info("data processed....")

    logging.info("Model creation.....")
    model = Word2Vec(
        sentences=processed_corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1
    )
    logging.info("Model created....")
    print(model.wv.most_similar(('ranveer')))
    model.save("word2vec.model")

if __name__ == "__main__":
    word2vec()
