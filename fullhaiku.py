import pyscipopt, cPickle
import numpy as np


class haikugenerator(pyscipopt.Model):
	def __init__(self,templates_f,T_f,s_f,t2n_f,w2n_f, pos_f,q2k_f,seed = None):
		super(haikugenerator,self).__init__()
		self.read_params(T_f,s_f,t2n_f,w2n_f, pos_f,q2k_f)
		self.choose_template(templates_f,seed)


	def read_params(self,T_f,s_f,t2n_f,w2n_f, pos_f,q2k_f):
		T = np.load(T_f)
		self.T = T/(T.sum(axis = 1, keepdims = True) +(T.sum(axis = 1, keepdims = True) == 0) )
		self.s = np.load(s_f)
		self.pos = np.load(pos_f)
		self.q2k = np.load(q2k_f)	
		self.t2n = cPickle.load(open(t2n_f,'rb'))
		self.w2n = cPickle.load(open(w2n_f,'rb'))
		
	def choose_template(self,templates_f,seed):
		np.random.seed(seed)
		template = np.random.choice(cPickle.load(open(templates_f,'rb')))
		seq_num = template['lengths']
		self.n1 = seq_num[0]
		self.n2 = seq_num[0]+seq_num[1]
		self.tagseq = template['tags']
		print self.n1, self.n2, self.tagseq

	def set_theme(self,q_list):
		n_terms = self.pos.shape[0]
		self.simvec = np.zeros(n_terms)
		for word in q_list:
			self.simvec[self.w2n[word]]+=1
		self.simvec = self.simvec.dot(self.q2k)
		self.simvec = self.simvec/np.linalg.norm(self.simvec)
		self.simvec = self.q2k.dot(self.simvec)
			
	def embed(self,w_list = []):
		h_len = self.x.shape[0]
		for word in w_list:
			self.addCons(pyscipopt.quicksum(self.x[i,self.w2n[word]] for i in xrange(h_len)) >= 1)


	def init_vars(self):
		n_terms, n_tags = self.pos.shape
		h_len = len(self.tagseq)
        	self.x = np.empty((h_len,n_terms),dtype ='object')
        	self.z = np.empty((h_len-1,n_terms,n_terms),dtype ='object')
        	self.dup = np.empty(n_terms, dtype = 'object')
		for i in xrange(n_terms):
			self.dup[i] = self.addVar(vtype = 'B')

		for i in xrange(h_len-1):
			for j in xrange(n_terms):
				self.x[i,j] = self.addVar(vtype = 'B')
				for k in xrange(n_terms):
					self.z[i,j,k] = self.addVar(vtype = 'B')

		for j in xrange(n_terms):
			self.x[h_len-1,j] = self.addVar(vtype = 'B')



	def struct_cons(self):
		n_terms, n_tags = self.pos.shape
		h_len = len(self.tagseq)

		#Each position in the haiku must be assigned exactly one word
		for i in xrange(h_len):
			self.addCons(pyscipopt.quicksum(self.x[i,j] for j in xrange(n_terms)) ==1)

		#each word must belong to POS indicated in the template
		for i in xrange(h_len):
			tg_idx = self.t2n[self.tagseq[i]]
			for j in xrange(n_terms):
				if self.pos[j,tg_idx] == 0:
					self.addCons(self.x[i,j] == 0)

		#stanzas must follow 5-7-5 structure
		self.addCons(pyscipopt.quicksum(self.s[j]*self.x[i,j] for i in range(0,self.n1) for j in range(n_terms)) == 5)
		self.addCons(pyscipopt.quicksum(self.s[j]*self.x[i,j] for i in range(self.n1,self.n2) for j in range(n_terms)) == 7)
		self.addCons(pyscipopt.quicksum(self.s[j]*self.x[i,j] for i in range(self.n2,h_len) for j in range(n_terms)) == 5)


		#upper bound on z ensures that it indicates whether xij = xi+1k = 1 in objective function
                for i in xrange(h_len-1):
                        for j in xrange(n_terms):
                                for k in xrange(n_terms):
                                        self.addCons(self.x[i,j] >= self.z[i,j,k])
                                        self.addCons(self.x[i+1,k] >= self.z[i,j,k])
		
		#lower bound on w_j ensures it is 1 if word j appears 2 or more times in resultant haiku
		rcp = 1.0/float(h_len) #reciprocal scaling factor
		for j in xrange(n_terms):
			self.addCons(self.dup[j] >=  pyscipopt.quicksum(rcp*self.x[i,j] for i in xrange(h_len))-rcp)

		
		#ad-hoc data preprocessing
	        for tok in ['\'','`','s']:
			if tok in self.w2n:
				for i in xrange(h_len):
					self.addCons(self.x[i,self.w2n[tok]]==0)


	def print_haiku(self):
		h_len,n_terms = self.x.shape
		sentence = []
		for i in xrange(h_len):
			for j in xrange(n_terms):
				if self.getVal(self.x[i,j])!=0:
					sentence.append(j)
					break
		n2w = {v:u for u,v in self.w2n.items()}
	        print ' '.join(n2w[k] for k in sentence[0:self.n1])
	        print ' '.join(n2w[k] for k in sentence[self.n1:self.n2])
	        print ' '.join(n2w[k] for k in sentence[self.n2:])



	def setObj(self,a,b,c):
		h_len, n_terms = self.x.shape
		ObjF = a*pyscipopt.quicksum(self.simvec[j]*self.x[i,j] for i in xrange(h_len) for j in xrange(n_terms))
		ObjF += b*pyscipopt.quicksum(self.T[j,k]*self.z[i,j,k] for i in xrange(h_len-1) for j in xrange(n_terms) for k in xrange(n_terms))
		ObjF -= c*pyscipopt.quicksum(self.dup[j] for j in xrange(n_terms))
        	self.setObjective(ObjF,'maximize')


	def init_model(self,q_list,w_list,a = 1,b = 4,c = 3,verbose = True):
		self.init_vars()
		self.struct_cons()
		self.set_theme(q_list)
		self.embed(w_list)	
		self.setObj(a,b,c)
		if verbose == False:
			self.hideOutput()

	def solve_model(self):
		self.optimize()
		self.print_haiku()


templates_f='templates.pickle';s_f = 's.npy';pos_f = 'pos.npy'
T_f = 'T.npy'; t2n_f = 't2n.pickle'; w2n_f = 'w2n.pickle'; q2k_f = 'q2k.npy'




mod = haikugenerator(templates_f,T_f,s_f,t2n_f,w2n_f, pos_f,q2k_f)

mod.init_model(['meadow'],['meadow'])
mod.solve_model()




