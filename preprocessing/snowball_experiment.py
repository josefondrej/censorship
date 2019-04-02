from nltk.stem import SnowballStemmer
from nltk.stem.snowball import EnglishStemmer

stemmer = SnowballStemmer("english")
stemmer = EnglishStemmer()

TEXT = "Once upon a time there lived in a certain village a little country girl, " + \
       "the prettiest creature who was ever seen. Her mother was excessively fond of her; " + \
       "and her grandmother doted on her still more. This good woman had a little red riding hood made for her. " + \
       "It suited the girl so extremely well that everybody called her Little Red Riding Hood."

stemmed_text = [stemmer.stem(word) for word in TEXT.split()]
print(stemmed_text)
