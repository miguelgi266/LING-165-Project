import numpy as np
import sklearn.decomposition as skd




def vocab(text_data):
	voc = list(set([tok for line in data for tok in line]))
	voc_dict = {voc[i]:i for i in range(len(voc))}
	return voc_dict
#def preprocess(text):


data = ['I like apples','Apples are good','a yellow btick road','tables are orange']
data = [d.split() for d in data]
voc_dict = vocab(data)
data = [[voc_dict[x] for x in d] for d in data]
n_docs = len(data)
n_terms = len(voc_dict)
tf_mat = np.zeros(shape = (n_docs,n_terms))


for i in range(len(data)):
	for j in range(len(data[i])):
		tf_mat[i][data[i][j]]+=1

#print tf_mat
#for x in voc_dict:
#	print x, voc_dict[x]

n_top = 3
lda = skd.LatentDirichletAllocation(n_top,learning_method = 'batch')

#print 'terms:', n_terms
#print 'docs: ',n_docs
#print 'topics: ',n_top


lda.fit(tf_mat)


#print lda.components_
#print lda.components_.shape
