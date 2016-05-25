from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem import SnowballStemmer
stemer=SnowballStemmer("english")
class StemmerVectorizer(CountVectorizer):
    def build_analyzer(self):
        bldr=super(StemmerVectorizer,self).build_analyzer()
        return lambda doc:(stemer.stem(word) for word in bldr(doc))
class StemmerTfidVec(TfidfVectorizer):
    def build_analyzer(self):
        builder=super(StemmerTfidVec,self).build_analyzer()
        return lambda doc:(stemer.stem(w) for w in builder(doc))
