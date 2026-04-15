from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from config.nltk_setup import download_nltk_resources

download_nltk_resources()

sentences = [
    "I visited Dominos in Bangalore and loved the pizza.",
    "The service at McDonald's was disappointing last Sunday.",
    "Ordered from Amazon and got delivery in 2 days.",
    "Had dinner at Taj Hotel in Mumbai, amazing experience!",
    "Flipkart customer support in India is very helpful."
]

for sentence in sentences:
    print("\nSentence:", sentence)
    tokens = word_tokenize(sentence)
    entities = ne_chunk(pos_tag(tokens))
    print(entities)