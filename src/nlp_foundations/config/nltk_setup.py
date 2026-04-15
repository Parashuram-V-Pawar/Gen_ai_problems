import nltk

def download_nltk_resources():
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger_eng',
        'words',
        'maxent_ne_chunker_tab'
    ]

    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)