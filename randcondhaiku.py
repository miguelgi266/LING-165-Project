import numpy as np
import pyscipopt as scip
import cPickle
import time

def rand_choice(seq):
	idx = np.random.randint(len(seq))
	return seq[idx]
#pos k for word j 
#x location i word j


def generate_sentence(tagseq,syllseq,T,clust,s,t2n,n2w,pos):
	max_len = len(tagseq)
	n_terms, n_clusts = clust.shape
	print 'nterms: %d, n_clusts: %d' %(n_terms,n_clusts)
	n_terms, ntags = pos.shape
        print 'nterms: %d, n_tags: %d' %(n_terms,ntags)
	w2n = {v:u for u,v in n2w.items()}
	print 'nterms:',len(w2n)
	n2t = {v:u for u,v in t2n.items()}
	print 'ntags:',len(n2t)
	words = ['brown','earth']
	q2k = np.load('q2k.npy')
	simvec = np.zeros(n_terms)
	for word in words:
        	simvec[w2n[word]]+=1
	simvec = q2k.dot(simvec.dot(q2k))
	print 'simvec shape:', simvec.shape

#	words = ['mother','picture']
#	words = ['shoes']
	model = scip.Model()#created IP model instance
	dup,x,y,z = var_init(model,max_len,n_terms,n_clusts)#initialize variables
	print 'setting constraints'
	struct_cons(model,dup,x,y,z,max_len,n_terms,s,syllseq,pos,t2n,tagseq,clust)#set structural constraints
	include_cons(model,x,words,w2n)

	print ' optimizing'

	### define objective function and optimize
#	ObjF = scip.quicksum(T[j,k]*z[i,j,k] for i in xrange(max_len-1) for j in xrange(n_clusts) for k in xrange(n_clusts)) \
	ObjF = scip.quicksum(simvec[j]*x[i,j] for i in xrange(max_len) for j in xrange(n_terms))- 1* scip.quicksum(dup[l] for l in xrange(n_terms))
#	ObjF = scip.quicksum(x[i,j] for i in xrange(max_len) for j in xrange(n_terms))
	model.setObjective(ObjF,'maximize')
	model.hideOutput()
	model.optimize()

	#print solution
	print_haiku(model,x,n2w,syllseq)


def var_init(model,max_len,n_terms,n_clusts):
	print 'number of variables:',max_len*(n_terms+n_clusts)+(max_len-1)*n_clusts*n_clusts
	print 'should not exceed:',max_len*200+(max_len-1)*200*200
        x = np.empty((max_len,n_terms),dtype ='object')# scip.scip.Variable) #ith position corresponds to word j
	y = np.empty((max_len,n_clusts),dtype = 'object')
        z = np.empty((max_len-1,n_clusts,n_clusts),dtype ='object')# scip.scip.Variable)
	dup = np.empty(n_terms,dtype = 'object')
	for i in xrange(n_terms):
		dup[i] = model.addVar(vtype = 'B') 
	for i in xrange(max_len):
                for j in xrange(n_terms):
                        x[i,j] = model.addVar(vtype = 'B')

		for j in xrange(n_clusts):
			y[i,j] = model.addVar(vtype = 'B')

	for i in xrange(max_len-1):
		for j in xrange(n_clusts):
			for k in xrange(n_clusts):
                                z[i,j,k] = model.addVar(vtype = 'B')


        for j in xrange(n_terms):
                x[max_len-1,j] = model.addVar(vtype = 'B')


	return dup,x,y,z 



def struct_cons(model,dup,x,y,z,max_len,n_terms,s,syllseq,pos,t2n,tagseq,clust):
	n_terms, n_clusts = clust.shape
        #every position must be associated to exactly one word
        for i in xrange(max_len):
                for k in xrange(n_clusts):
                        model.addCons(scip.quicksum(clust[j,k]*x[i,j] for j in range(n_terms)) == y[i,k])


        for i in xrange(max_len):
                model.addCons(scip.quicksum(x[i,j] for j in xrange(n_terms)) ==1)
	
	#every words associated to a cluster###
#	for i in xrange(max_len):
#		for k in xrange(n_clusts):
#			model.addCons(scip.quicksum(clust[j,k]*x[i,j] for j in range(n_terms)) == y[i,k])
#			model.addCons(scip.quicksum(clust[j,k]*x[i,j] for j in range(n_terms)) >= y[i,k])
#			model.addCons(scip.quicksum(clust[j,k]*x[i,j] for j in range(n_terms)) <= y[i,k])
	#accounts for transition from ij to i+1k
        for i in xrange(max_len-1):
                for j in xrange(n_clusts):
                        for k in xrange(n_clusts):
                                model.addCons(y[i,j] >= z[i,j,k])
                                model.addCons(y[i+1,k] >= z[i,j,k])

	#for term penalty
	for l in xrange(n_terms):
		for p in xrange(max_len):
			for q in xrange(max_len):
				if p!=q:
					model.addCons(dup[l] >= x[p,i]+x[q,l] -1)
					

	#syllabic constraints for stanzas
        model.addCons(scip.quicksum(s[j]*x[i,j] for i in xrange(0,syllseq[0]) for j in xrange(n_terms)) == 5)
        model.addCons(scip.quicksum(s[j]*x[i,j] for i in xrange(syllseq[0],syllseq[0]+syllseq[1]) for j in xrange(n_terms)) == 7)
        model.addCons(scip.quicksum(s[j]*x[i,j] for i in xrange(syllseq[0]+syllseq[1],max_len) for j in xrange(n_terms)) == 5)
	
	#haiku must follow specified tag sequence
	print len(tagseq)
	print tagseq
        for i in range(max_len):
		print i, tagseq[i]
#		print tagseq[i], t2n[tagseq[i]]
		model.addCons(scip.quicksum(x[i,j]* pos[j,t2n[tagseq[i]]] for j in range(n_terms)) == 1)







def print_haiku(model,x,n2w,syllseq):
	max_len,n_terms = x.shape
	sentence = []
        for i in xrange(max_len):
                for j in xrange(n_terms):
                        if model.getVal(x[i,j])!=0:
#                                print 'position %d, word %s' %(i,n2w[j])
                                sentence.append(j)

#       print ' '.join([n2w[k] for k in sentence])
        print ' '.join(n2w[k] for k in sentence[0:syllseq[0]])
        print ' '.join(n2w[k] for k in sentence[syllseq[0]:syllseq[0]+syllseq[1]])
        print ' '.join(n2w[k] for k in sentence[syllseq[0]+syllseq[1]:])


def include_cons(model,x,words,w2n):
	max_len, n_terms = x.shape
	for w in words:
		print 'must include %s' %w
		model.addCons(scip.quicksum(x[i,w2n[w]] for i in xrange(max_len)) >= 1 ) 



seq = np.random.choice(cPickle.load(open('templates.pickle')))
s = np.load('s.npy')
pos = np.load('pos.npy')
T = np.load('newT.npy')
clust = np.load('clust.npy')
t2n= cPickle.load( open('t2n.pickle','rb'))
w2n = cPickle.load(open('w2n.pickle','rb'))
n2w = {v:u for u,v in w2n.items()}
#print 'vocab size:', len(w2n)
syllseq = seq['lengths']
tagseq =  seq['tags']
generate_sentence(tagseq,syllseq,T,clust,s,t2n,n2w,pos)

