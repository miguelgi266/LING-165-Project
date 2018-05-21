import numpy as np
import pyscipopt as scip
import cPickle
import time

def rand_choice(seq):
	idx = np.random.randint(len(seq))
	return seq[idx]
#pos k for word j 
#x location i word j


def generate_sentence(T,s,t2n,n2w,pos,syll_count,max_len):
        tagseq = ['DT', 'JJ', 'NN', 'NNS', 'VBZ', 'NN', 'VBN', 'IN', 'DT', 'NN', 'IN', 'JJ', 'NN']
        syllseq  = [4,5,4]
	n_terms, ntags = pos.shape
	w2n = {v:u for u,v in n2w.items()}
	words = ['baby']
	model = scip.Model()#created IP model instance
	x,z = var_init(model,max_len,n_terms)#initialize variables
	print 'set up constraints'

	struct_cons(model,x,z,max_len,n_terms,s,syllseq,pos,t2n,tagseq)#set structural constraints
	include_cons(model,x,words,w2n)

	print 'begin optimization'

	### define objective function and optimize
	ObjF = scip.quicksum(T[j,k]*z[i,j,k] for i in range(0,max_len-1) for j in range(n_terms) for k in range(n_terms))
	model.setObjective(ObjF,'maximize')
	model.hideOutput()
	model.optimize()

	#print solution
	print_haiku(model,x,n2w,syllseq)


def var_init(model,max_len,n_terms):
        x = np.empty((max_len,n_terms),dtype ='object')# scip.scip.Variable) #ith position corresponds to word j
        z = np.empty((max_len-1,n_terms,n_terms),dtype ='object')# scip.scip.Variable)
        for i in xrange(max_len-1):
                for j in xrange(n_terms):
                        x[i,j] = model.addVar(vtype = 'B')
                        for k in xrange(n_terms):
                                z[i,j,k] = model.addVar(vtype = 'B')

        for j in xrange(n_terms):
                x[max_len-1,j] = model.addVar(vtype = 'B')
	return x,z 



def struct_cons(model,x,z,max_len,n_terms,s,syllseq,pos,t2n,tagseq):
        #every position must be associated to exactly one word
        for i in xrange(max_len):
                model.addCons(scip.quicksum(x[i,j] for j in xrange(n_terms)) ==1)
	

	#accounts for transition from ij to i+1k
        for i in xrange(max_len-1):
                for j in xrange(n_terms):
                        for k in xrange(n_terms):
                                model.addCons(x[i,j] >= z[i,j,k])
                                model.addCons(x[i+1,k] >= z[i,j,k])

	#syllabic constraints for stanzas
        model.addCons(scip.quicksum(s[j]*x[i,j] for i in xrange(0,syllseq[0]) for j in xrange(n_terms)) == 5)
        model.addCons(scip.quicksum(s[j]*x[i,j] for i in xrange(syllseq[0],syllseq[0]+syllseq[1]) for j in xrange(n_terms)) == 7)
        model.addCons(scip.quicksum(s[j]*x[i,j] for i in xrange(syllseq[0]+syllseq[1],max_len) for j in xrange(n_terms)) == 5)
	
	#haiku must follow specified tag sequence
        for i in range(max_len):
		model.addCons(scip.quicksum(x[i,j]* pos[j,t2n[tagseq[i]]] for j in range(n_terms)) == 1)


def print_haiku(model,x,n2w,syllseq):
	max_len,n_terms = x.shape
	sentence = []
        for i in range(max_len):
                for j in range(n_terms):
                        if model.getVal(x[i,j])!=0:
                                print 'position %d, word %s' %(i,n2w[j])
                                sentence.append(j)

#       print ' '.join([n2w[k] for k in sentence])
        print ' '.join(n2w[k] for k in sentence[0:syllseq[0]])
        print ' '.join(n2w[k] for k in sentence[syllseq[0]:syllseq[0]+syllseq[1]])
        print ' '.join(n2w[k] for k in sentence[syllseq[0]+syllseq[1]:])


def include_cons(model,x,words,w2n):
	max_len, n_terms = x.shape
	for w in words:
		model.addCons(scip.quicksum(x[i,w2n[w]] for i in xrange(max_len)) >= 1 ) 




s = np.load('s.npy')
pos = np.load('pos.npy')
T = np.load('T.npy')
t2n= cPickle.load( open('t2n.pickle','rb'))
w2n = cPickle.load(open('w2n.pickle','rb'))
n2w = {v:u for u,v in w2n.items()}
print 'vocab size:', len(w2n)


generate_sentence(T,s,t2n,n2w,pos,17,13)

