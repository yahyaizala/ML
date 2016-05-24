import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
data=sp.genfromtxt("web_traffic.tsv",delimiter="\t")
def error(f,x,y):
    return sp.sum((f(x)-y)**2)
x=data[:,0]
y=data[:,1]
#print sp.sum(sp.isnan(y)) #8
x=x[~sp.isnan(y)]
y=y[~sp.isnan(y)]
fp1,residuals,rank,sv,srand=sp.polyfit(x,y,1,full=True)
print "Model params {}".format(fp1)
print "Residuals {}".format(residuals)
f1=sp.poly1d(fp1)
#print error(f1,x,y)
fx=sp.linspace(0,x[-1],1000)
plt.plot(fx,f1(fx),linewidth=4)
plt.legend(["d=%i"%f1.order],loc="upper left")
#----- 2 dim func---
f2p=sp.polyfit(x,y,10)
f2=sp.poly1d(f2p)

plt.plot(fx,f2(fx),linewidth=6)
'''
print f2
print error(f2,x,y)

Model params [   2.59619213  989.02487106]
Residuals [  3.17389767e+08]

           10             9             8             7             6
-3.74e-22 x  + 1.365e-18 x - 2.143e-15 x + 1.899e-12 x - 1.046e-09 x
              5             4           3          2
 + 3.709e-07 x - 8.456e-05 x + 0.01192 x - 0.9416 x + 33.37 x + 1264
 
121942326.364

'''

print sp.sum(sp.isnan(y)) #0
plt.scatter(x,y,s=10)
plt.title("Web Traffic Over Last Month")
plt.xlabel("Time")
plt.ylabel("Hits/Hours")
plt.autoscale(tight=True)
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.grid(True,linestyle="-",color="0.3")
plt.show()
