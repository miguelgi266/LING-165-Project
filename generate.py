import sklearn.decomposition as skd
import numpy as np
import pyscipopt as scip
import cPickle

max_len = 16
##########################
#Import model parameters##
A = np.load('TRM.npy')#hidden state transitions A[s1][s2] = P(s2|s1)
B = np.load('S2O.npy')#emission probabilities B[s][o] = P(O|S)
with open('ldamodel.pickle') as f: #lda topic model of data
	lda = cPickle.load(f) 
with open('key.pickle','r') as f:
	w2n = cPickle.load(f)
n2w = {v:u for u,v in w2n.items()}


n_top,n_docs = lda.components_.shape
n_stt, n_terms = B.shape





####################
#choose sentence 
sent = ['coke', 'is', 'a', 'type', 'of', 'soda']



if len(sent)>max_len-2:
	print 'too long'
	exit()
sent = ['<s>']+sent
while len(sent)!= max_len: sent.append('</s>')
sent = [n2w[tok] for tok in sent]



print sent




model = scip.Model()

###############initialize variables###############################



x = np.empty((max_len,n_stt,n_stt),dtype='object') #P(state_i|state_i-1)
v = np.empty((max_len,n_stt,n_terms),dtype = 'object') #P(emission_i|state_i)
y= np.empty((max_len,n_stt),dtype='object') #emission for word i
z = np.array([model.addVar(vtype ='B') for i in range(n_stt)])
for i in range(max_len):
        for j in range(n_stt):
                y[i][j] = model.addVar(vtype='B')#name variables later??
                for k in range(n_stt):
                        x[i][j][k] = model.addVar(vtype = 'B')#name variables later?? 
                for k in range(n_terms):
                        v[i][j][k] = model.addVar(vtype = 'B')	
		



################Set Constraints####################################

#for the first word, make sure the previous tag is <s>
s_0 = 1#-1#states.index('<s>')
for j in range(n_stt):
        if j ==s_0:
                continue
        for k in range(n_stt):
                model.addCons(x[0][j][k] == 0)

#for the last word, make sure the next tag is </s>
s_f = states.index('</s>')



#Cannot reach z[j] = 0 for states q where A_M[q,s_f] = 1
for j in range(n_stt):
        if A_M[j][s_f] == 1:
                model.addCons(z[j] == 0)

#z_j must have initial state of yij 
for j in range(n_stt):
        if j != s_f:
                model.addCons(y[max_len-1,j] - z[j] ==0)#z[j]*A[j,</s>]
#z_j must be 1 somewhere
model.addCons(scip.quicksum(z[i] for i in range(max_len)) == 1)



#only one state transition to a word
for i in range(max_len):
        model.addCons(scip.quicksum(x[i][j][k] for j in range(n_stt) for k in range(n_stt)) == 1)


#make sure endpoint of state transition matches state of word
#make sure endpoint of state transition matches state of word
for i in range(max_len):
        for j in range(n_stt):
                model.addCons(y[i][j]-scip.quicksum(x[i][k][j] for k in range(n_stt))==0)
for i in range(max_len):
        model.addCons(scip.quicksum(y[i][j] for j in range(n_stt)) == 1)

#make sure starting point of state transition for word i
#has the same previous state as word i-1
for i in range(1,max_len):
        for j in range(n_stt):
                model.addCons(y[i-1][j] -scip.quicksum(x[i][j][k] for k in range(n_stt)) == 0)


#zero out values that are not possible
for i in range(max_len):
        for j in range(n_stt):
                for k in range(n_stt):
                        if A_M[j][k] == 1:
                                model.addCons(x[i][j][k] == 0)
                for k in range(n_terms):
                        if B_M[j][k] == 1:
                                model.addCons(v[i][j][k] == 0)

#only one emission per word
for i in range(max_len):
        model.addCons(scip.quicksum(v[i][j][k] for j in range(n_stt) for k in range(n_terms)) ==1)

#emission must be from current state
for i in range(max_len):
        for j in range(n_stt):
                model.addCons(y[i][j]- scip.quicksum(v[i][j][k] for k in range(n_terms)) == 0)
#emission limited to word
for i in range(max_len):
        K = emission.index(sent[i])
        for j in range(n_stt):
                for k in range(n_terms):
                        if k == K:
                                continue
                        model.addCons(v[i][j][k] ==0)


###############Objective Function##########################
##Fix objective function by adding coefficients
model.setObjective(scip.quicksum(A_M[j][k]*x[i][j][k] for i in range(max_len) for j in range(n_stt) for k in range(n_stt)) +\
        scip.quicksum(B_M[j][k]*v[i][j][k] for i in range(max_len) for j in range(n_stt) for k in range(n_terms)) +\
        scip.quicksum(A_M[j][s_f]*z[j] for j in range(n_stt)),'maximize') +\
	scip.quicksum(T_M[k][j]*y[j][i] for i in range(n_stt) for j in range(max_len) for k in range(n_top))

model.hideOutput()
model.optimize()
print model.getObjVal()

print 'states'
for i in range(max_len):
        for j in range(n_stt):
                if model.getVal(y[i][j]) != 0:
                        print sent[i],states[j]

print 'transitions'
for i in range(max_len):
        for j in range(n_stt):
                for k in range(n_stt):
                        if model.getVal(x[i][j][k])!=0:
                                print 'from:',states[j],'\tto',states[k],A_M[j][k]

print 'emissions'
for i in range(max_len):
        for j in range(n_stt):
                for k in range(n_terms):
                        if model.getVal(v[i][j][k])!=0:
                                print 'state:',states[j],'\tem',emission[k],B_M[j][k]

print 'into last state'
for i in range(n_stt):
        if model.getVal(z[i])!=0:
                print states[i]



    	
