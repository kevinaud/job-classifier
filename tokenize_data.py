import pandas as pd
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
	return text.split()
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]
