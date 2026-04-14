import logging
from gensim.models import Word2Vec
model = Word2Vec.load("word2vec.model")

logging.basicConfig(level=logging.INFO)
positive_words = ['careerbest', 'good', 'strong', 'engaging', 'intense']
negative_words = ['slightly', 'slow', 'tighter']

def similarity_test():
    logging.info("Positive vs Positive:")
    for i in range(len(positive_words)):
        for j in range(i+1, len(positive_words)):
            w1, w2 = positive_words[i], positive_words[j]
            print(f"{w1} vs {w2}: {model.wv.similarity(w1, w2):.3f}")
    print()

    logging.info("\nNegative vs Negative:")
    for i in range(len(negative_words)):
        for j in range(i+1, len(negative_words)):
            w1, w2 = negative_words[i], negative_words[j]
            print(f"{w1} vs {w2}: {model.wv.similarity(w1, w2):.3f}")
    print()

    logging.info("\nPositive vs Negative:")
    for p in positive_words:
        for n in negative_words:
            print(f"{p} vs {n}: {model.wv.similarity(p, n):.3f}")
    print()

if __name__ == "__main__":
    similarity_test()