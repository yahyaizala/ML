from sklearn.datasets import fetch_20newsgroups
from Utils import  *
from sklearn.cluster import KMeans
groups=['comp.graphics','comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
        'comp.windows.x', 'sci.crypt']

train,test=fetch_20newsgroups(subset="train",categories=groups),fetch_20newsgroups(subset="test",categories=groups)
vectorizer=StemmerTfidVec(min_df=1,stop_words="english",max_df=0.5,decode_error="ignore")
vectorized=vectorizer.fit_transform(train.data)
km=KMeans(n_clusters=10,init="random",n_init=1,verbose=0,random_state=3)
km.fit(vectorized)
prbl="""Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks."""

new=vectorizer.transform([prbl])
new_label=km.predict(new)[0]
similiar_i=(km.labels_==new_label).nonzero()[0]

'''
print similiar_i
[  13   30   39   50   54   75   79   82   86  116  117  126  130  132  185
  208  227  231  242  259  307  319  322  324  367  377  385  394  410  414
  426  462  471  484  489  501  525  529  531  557  567  581  601  650  656
  687  690  742  773  803  807  831  859  910  940  950  999 1112 1119 1173
 1187 1193 1199 1223 1234 1282 1286 1315 1371 1388 1481 1500 1521 1525 1531
 1535 1559 1565 1592 1598 1631 1634 1660 1676 1687 1696 1706 1744 1759 1760
 1764 1772 1774 1796 1808 1905 1939 1978 2009 2035 2064 2067 2079 2082 2105
 2109 2129 2140 2146 2154 2189 2238 2271 2302 2306 2352 2439 2449 2459 2473
 2526 2556 2561 2580 2587 2649 2680 2702 2752 2763 2813 2830 2851 2861 2907
 2940 2943 2945 2946 2963 2970 3025 3036 3039 3076 3108 3126 3179 3182 3212
 3241 3249 3279 3312 3314 3359 3366 3375 3392 3405 3423 3456 3463 3494 3506
 3522 3528]
'''
similar=[]
for i in similiar_i:
    dist=sp.linalg.norm((new-vectorized[i]).toarray())
    similar.append((dist,train.data[i]))
print similar[0]
print similar[len(similar)/10]
print similar[len(similar)/2]
'''
(1.2432567747385816,
 u'From: djweisbe@unix.amherst.edu (David Weisberger)\nSubject: Booting from
 B drive\nNntp-Posting-Host: amhux3.amherst.edu\nOrganization: large\nX-Newsreader:
IN [version 1.1 PL7]\nLines: 17\n\nI have a 5 1/4" drive as drive A.
How can I make the system boot from\nmy 3 1/2" B drive?
 (Optimally, the computer would be able to boot\nfrom either A or B, checking them in order for a bootable disk.
But\nif I have to switch cables around and simply switch the drives so that\nit can\'t boot 5 1/4" disks, that\'s OK.
 Also, boot_b won\'t do the trick\nfor me.)\n\nThanks,\n  Davebo\n\n--\nDavid Weisberger   | Q: Mr. President, do you care to say any more about the\n                   |    operational details of the airlift?\ndjweisbe           | THE PRESIDENT:  No.\n@unix.amherst.edu  | Q: How about explaining to the American people why it\'s\n                   |    an important issue for the United States to undertake?\n                   | THE PRESIDENT:  What?\n')
(1.3581923510020415, u'From: balog@eniac.seas.upenn.edu (Eric J Balog)\nSubject: SWITCH 3.5" TO A:?\nOrganization: University of Pennsylvania\nLines: 39\nNntp-Posting-Host: eniac.seas.upenn.edu\n\nHi!\n\nI\'d like to switch my floppy drives so that my 3.5" b: drive becomes a:, while\nmy 5.25" a: becomes b:. I\'m having a few problems, though.\n\nI know that the ribbon cable must be switched, as well as the CMOS settings, \nto reflect this change, and I think that I\'ve done that correctly. However, the\ndrives do not operate correctly in this configuration. From the C:> prompt, if \nI type a:, the 5.25" drive light comes on; if I type b:, both the light for the\n5.25" and 3.5" drives come on.\n\nThere are some jumpers on each drive:\n5.25"  Label   Original Pos.   Pos. I changed it to\n        DS0      ON               OFF\n\tDS1\t OFF\t\t  ON\n\tDS2\t ON\t\t  ON\n\tDS3\t OFF\t\t  OFF\n\tIO\t OFF\t\t  OFF\n\tMS1\t OFF\t\t  OFF\n\tD-R\t ON \t\t  ON\n\tMS2\t ON\t\t  ON\n\tFG\t OFF\t\t  OFF\n\n3.5"    DS0\t OFF\t\t  ON\n\tDS1\t ON\t\t  OFF\n\tDS2\t OFF\t\t  OFF\n\tDS3\t OFF\t\t  OFF\n\tMM\t ON\t\t  ON\n\tDC\t ON\t\t  ON\n\tMD\t OFF\t\t  OFF\n\tTTL/C-MO8 ON\t\t  ON\n\n\nAny help or suggestions would be greatly appreciated.\n\nThanks in advance.\n\nEric Balog\nbalog@eniac.seas.upenn.edu\n')
(1.4035615974804785, u'From: jim@jagubox.gsfc.nasa.gov (Jim Jagielski)\nSubject: Re: Quadra SCSI Problems???\nKeywords: Quadra SCSI APS\nLines: 29\nReply-To: jim@jagubox.gsfc.nasa.gov (Jim Jagielski)\nOrganization: NASA/Goddard Space Flight Center\n\ntzs@stein2.u.washington.edu (Tim Smith) writes:\n\n>> ATTENTION: Mac Quadra owners: Many storage industry experts have\n>> concluded that Mac Quadras suffer from timing irregularities deviating\n>> from the standard SCSI specification. This results in silent corruption\n>> of data when used with some devices, including ultra-modern devices.\n>> Although I will not name the devices, since it is not their fault, an\n>> example would be a Sony 3.5 inch MO, without the special "Mac-compatible"\n>> firmware installed. One solution, sometimes, is to disable "blind writes"\n\n>This doesn\'t sound right to me.  Don\'t Quadras use the 53C96?  If so, the\n>Mac has nothing to do with the SCSI timing.  That\'s all handled by the\n>chip.  About the only the timing could be wrong is if Apple programs the\n>clock registers wrong on the 96.  That, however, should only really hurt\n>synchronous transfer, which is not used by the Mac SCSI Manager.\n\n>Furthermore, disabling blind writes should be meaningless on a Quadra.\n>On Macs that used the 5380, which is a much lower level SCSI chip, the\n>Mac was responsible for the handshake of each byte transferred.  Blind\n>mode affected how the Mac handled that handshake.  On the 5396, the\n>handshake is entirely handled by the chip.\n\nThe docs say that it\'s a SCSI Manager bug, if this changes things at all...\n-- \n    Jim Jagielski               |  "And he\'s gonna stiff me. So I say,\n    jim@jagubox.gsfc.nasa.gov   |   \'Hey! Lama! How about something,\n    NASA/GSFC, Code 734.4       |   you know, for the effort!\'"\n    Greenbelt, MD 20771         |\n\n')


'''



'''
print km.labels_
print km.labels_.shape
Initialization complete
Iteration  0, inertia 6708.907
Iteration  1, inertia 3424.690
Iteration  2, inertia 3411.498
Iteration  3, inertia 3406.046
Iteration  4, inertia 3401.780
Iteration  5, inertia 3399.419
Iteration  6, inertia 3398.293
Iteration  7, inertia 3397.813
Iteration  8, inertia 3397.549
Iteration  9, inertia 3397.331
Iteration 10, inertia 3397.272
Iteration 11, inertia 3397.174
Iteration 12, inertia 3397.048
Iteration 13, inertia 3396.904
Iteration 14, inertia 3396.766
Iteration 15, inertia 3396.610
Iteration 16, inertia 3396.498
Iteration 17, inertia 3396.438
Iteration 18, inertia 3396.329
Iteration 19, inertia 3396.248
Iteration 20, inertia 3396.227
Iteration 21, inertia 3396.206
Iteration 22, inertia 3396.178
Iteration 23, inertia 3396.154
Iteration 24, inertia 3396.115
Iteration 25, inertia 3396.098
Iteration 26, inertia 3396.057
Iteration 27, inertia 3396.000
Iteration 28, inertia 3395.980
Iteration 29, inertia 3395.934
Iteration 30, inertia 3395.856
Iteration 31, inertia 3395.824
Iteration 32, inertia 3395.728
Iteration 33, inertia 3395.683
Iteration 34, inertia 3395.609
Iteration 35, inertia 3395.465
Iteration 36, inertia 3395.342
Iteration 37, inertia 3395.139
Iteration 38, inertia 3394.860
Iteration 39, inertia 3394.667
Iteration 40, inertia 3394.413
Iteration 41, inertia 3393.964
Iteration 42, inertia 3393.537
Iteration 43, inertia 3393.520
Converged at iteration 43
[6 0 5 ..., 9 8 7]
(3531L,)

'''




'''
data=fetch_20newsgroups(subset="all")
print data.target_names


['alt.atheism', 'comp.graphics',
'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
  'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
   'talk.politics.misc', 'talk.religion.misc']

'''
