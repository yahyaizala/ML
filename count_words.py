from __future__ import division
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import scipy as sp
import sys
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

def calcDist(v1,v2):
    mult=v2-v1
    return sp.linalg.norm(mult.toarray())
def calcDistNormalized(v1,v2):
    v1=v1/sp.linalg.norm(v1.toarray())
    v2=v2/sp.linalg.norm(v2.toarray())
    dels=v2-v1
    return sp.linalg.norm(dels.toarray())
def execute(countVectorizer,train):
    new = "databases images"
    newPv = countVectorizer.transform([new])
    bestDoc = None
    dist = sys.maxint
    bestI = None
    for a in range(sample):
        p = stuf[a]
        if p == new: continue
        posVec = train.getrow(a)
        d = calcDistNormalized(posVec, newPv)
        print">>> post %i with distance =% .2f : %s" % (a, d, p)
        if d < dist:
            dist = d
            bestI = a
    print">>> Best  post is %i with distance =%.2f" % (bestI, dist)

'''
def tfidf(t, d, D):
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    idf = sp.log(float(len(D)) / (len([doc for doc in D if t in doc])))
    return tf * idf
a,abb,abc=["a"],["a","b","b"],["a","b","c"]
corpus=[a,abb,abc]
print tfidf("b",abb,corpus)

'''

dir="data/toy"
stuf=[open(os.path.join(dir,f)).read() for f in os.listdir(dir)]
viTfid=StemmerTfidVec(min_df=1,stop_words="english",decode_error="ignore")
train=viTfid.fit_transform(stuf)
sample,feature=train.shape
execute(viTfid,train)
print("*"*100)
countVectorizer=StemmerVectorizer(min_df=1,stop_words="english",decode_error="ignore")
train=countVectorizer.fit_transform(stuf)
sample,feature=train.shape
execute(countVectorizer,train)
#print train.shape
#(5,25)
#print countVectorizer.get_feature_names()
#[u'about', u'actually', u'capabilities', u'contains', u'data',
# u'databases', u'images', u'imaging', u'interesting',
#  u'is', u'it', u'learning', u'machine', u'most', u'much',
# u'not', u'permanently', u'post', u'provide', u'save',
# u'storage', u'store', u'stuff', u'this', u'toy']


'''
     print posVec,"---"
    0, 7)	1
  (0, 5)	1
  (0, 18)	1
  (0, 20)	1
  (0, 2)	1 ---
  (0, 7)	1
  (0, 5)	1
  (0, 13)	1
  (0, 19)	1
  (0, 6)	1
  (0, 16)	1 ---
'''




'''
print newPv
print newPv.toarray()

------
(0, 5)	1
(0, 6)	1

[[0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]


'''

'''
print train
print train.toarray().T
  (0, 23)	1
  (0, 9)	1
  (0, 24)	1
  (0, 17)	1
  (0, 0)	1
  (0, 12)	1
  (0, 11)	1
  (0, 1)	1
  (0, 10)	1
  (0, 3)	1
  (0, 15)	1
  (0, 14)	1
  (0, 8)	1
  (0, 22)	1
  (1, 7)	1
  (1, 5)	1
  (1, 18)	1
  (1, 20)	1
  (1, 2)	1
  (2, 7)	1
  (2, 5)	1
  (2, 13)	1
  (2, 19)	1
  (2, 6)	1
  (2, 16)	1
  (3, 7)	1
  (3, 5)	1
  (3, 21)	1
  (3, 4)	1
  (4, 7)	3
  (4, 5)	3
  (4, 21)	3
  (4, 4)	3

[[1 0 0 0 0]
 [1 0 0 0 0]
 [0 1 0 0 0]
 [1 0 0 0 0]
 [0 0 0 1 3]
 [0 1 1 1 3]
 [0 0 1 0 0]
 [0 1 1 1 3]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [0 0 1 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [0 0 1 0 0]
 [1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 1 0 0 0]
 [0 0 0 1 3]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]]


'''
'''
>>> post 0 with distance = 1.41 : This is a toy post about machine learning. Actually, it contains not much interesting stuff.
>>> post 1 with distance = 1.08 : Imaging databases provide storage capabilities.
>>> post 2 with distance = 0.86 : Most imaging databases save images permanently.

>>> post 3 with distance = 0.92 : Imaging databases store data.
>>> post 4 with distance = 0.92 : Imaging databases store data. Imaging databases store data. Imaging databases store data.
>>> Best  post is 2 with distance =0.86
****************************************************************************************************
>>> post 0 with distance = 1.41 : This is a toy post about machine learning. Actually, it contains not much interesting stuff.
>>> post 1 with distance = 0.86 : Imaging databases provide storage capabilities.
>>> post 2 with distance = 0.63 : Most imaging databases save images permanently.

>>> post 3 with distance = 0.77 : Imaging databases store data.
>>> post 4 with distance = 0.77 : Imaging databases store data. Imaging databases store data. Imaging databases store data.
>>> Best  post is 2 with distance =0.63


'''
