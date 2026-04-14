import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config.nltk_setup import download_nltk_resources

download_nltk_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

reviews = [
    "Ranveer Singh delivers a career-best performance.",
    "The film is intense, gripping, and keeps you hooked with its raw storytelling and strong emotional scenes.",
    "Good concept, but the pacing feels slow at times.",
    "Action-packed and engaging, though slightly long.",
    "Strong visuals, but the story could be tighter."
]

def preprocess(data = reviews):
    cleaned = [clean_text(review) for review in data]
    tokens = [word_tokenize(text) for text in cleaned]
    filtered = [[word for word in review if word not in stop_words] for review in tokens]
    lemmatized = [[lemmatizer.lemmatize(w) for w in review] for review in filtered]
    return cleaned, tokens, filtered, lemmatized

if __name__=="__main__":
    cleaned, tokens, filtered, lemmatized = preprocess()
    print("\n")
    print(f"\nOriginal:\n{reviews}")
    print(f"\nCleaned:\n{cleaned}")
    print(f"\nTokens:\n{tokens}")
    print(f"\nFiltered:\n{filtered}")
    print(f"\nLemmatized:\n{lemmatized}")