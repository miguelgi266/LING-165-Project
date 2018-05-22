import cPickle, scipy.cluster
import numpy as np
import nltk.cluster#.kmeans
def l1(x,y):
	return np.abs(x-y).sum()
T = np.load('T.npy')
print 'n_rows:', T.shape[0]
print 'n_unique:',len(set([tuple(T[i]) for i in range(T.shape[0])]))

#T2 = (T==0).astype('float64')
u, s, v = np.linalg.svd(T)
c = min(len(s),T.shape[0]/2,150)
print c
T2 = u[:,:c].dot(np.diag(s[:c]))
del u,s,v

n_clust = 50#T.shape[0]
dist,clust = scipy.cluster.vq.kmeans2(T2,n_clust,iter = 15,minit = 'points')
#kclusterer = nltk.cluster.kmeans.KMeansClusterer(n_clust,distance = l1,repeats = 20)
#clust = kclusterer.cluster(T2)
with open('w2n.pickle') as f:
	w2n = cPickle.load(f)

n2w = {u:v for v,u in w2n.items()}

d = {}

for i in range(clust.shape[0]):
	if clust[i] in d:
		d[clust[i]].append(i)
	else:
		d[clust[i]] = []
		d[clust[i]].append(i)

#for i in d:
#	print d[i]	

clase = np.zeros((T.shape[0],n_clust))

for lbl in d:
	for inst in d[lbl]:
		clase[inst,lbl]+=1


T =  clase.T.dot(T.dot(clase))
np.save('newT.npy',T)
np.save('clust.npy',clase)
