import logging
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize
from config.nltk_setup import download_nltk_resources

w2v_model = Word2Vec.load("word2vec.model")

logging.basicConfig(level=logging.INFO)

reviews = [
    "Ranveer Singh delivers a career-best performance.",
    "The film is intense, gripping, and keeps you hooked with its raw storytelling and strong emotional scenes.",
    "Good concept, but the pacing feels slow at times.",
    "Action-packed and engaging, though slightly long.",
    "Strong visuals, but the story could be tighter."
]

def compare_similarity():
    processed = [word_tokenize(text.lower()) for text in reviews]

    ft_model = FastText(
        sentences=processed,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    print("Word2Vec similarity:")
    print(w2v_model.wv.similarity("ranveer", "performance"))

    print("\nFastText similarity:")
    print(ft_model.wv.similarity("ranveer", "performance"))

    print("\nFastText handles unseen word:")
    print(ft_model.wv["feeling"])
    try:
        print(w2v_model.wv["feeling"])
    except Exception as e:
        print("\nWord2Vec error:", e)

    print("\nWord2Vec:")
    print(w2v_model.wv.most_similar("strong"))

    print("\nFastText:")
    print(ft_model.wv.most_similar("strong"))

if __name__ == "__main__":
    compare_similarity()