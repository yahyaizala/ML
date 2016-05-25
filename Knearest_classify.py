from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
data=load_iris()
features=data.data
target=data.target
target_names=data.target_names
labels=target_names[target]
clf=KNeighborsClassifier(n_neighbors=1)
from sklearn.cross_validation import KFold
kfold=KFold(len(features),n_folds=5,shuffle=True)
means=[]
cnt=0
for tb,tst in kfold:
    clf.fit(features[tb],labels[tb])
    print  tb,tst
    print features[tb]
    print  labels[tb]
    cnt +=1
    print cnt,"---------"*10
    pred=clf.predict(features[tst])
    print pred,"---"*12
    curmean=np.mean(pred==labels[tst])
    means.append(curmean)
print means
print np.mean(means)