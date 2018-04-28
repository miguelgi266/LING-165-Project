import collections, random, math, cPickle
import sklearn.decomposition as skd
import time
import numpy as np
###Objective Function
def n_gram(x,n):
	return(zip(*[x[i:] for i in range(n)]))
def score(texto,tagdict):

	tagtext = [[tagdict[x] for x in sent] for sent in texto ]
	ngram_freq = collections.Counter([x for sent in tagtext for x in n_gram(sent,2)])
#	for x in ngram_freq: print x, ngram_freq[x]
	wc_freq = collections.Counter([ x for line in zip(texto,tagtext) for x in zip(line[0],line[1]) ])
#	for x in wc_freq: print wc_freq[x], x
	tag_freq = collections.Counter([x for sent in tagtext for x in sent])
#	for x in tag_freq: print x, tag_freq[x]
	
	val = sum([sum([math.log(ngram_freq[(tagtext[i][j-1],tagtext[i][j])]*wc_freq[(texto[i][j],tagtext[i][j])]) for j in range(1,len(tagtext[i]))]) for i in range(len(tagtext))])	
	val = val - sum([sum([math.log(tag_freq[tagtext[i][j-1]]*tag_freq[tagtext[i][j]]) for j in range(1,len(tagtext[i]))]) for i in range(len(tagtext))])
#	print val
	return (val)





class  population:
	
###########maybe create an individuals class in the future
###########it would contain parameters describing individual as well as the fitness
	members = []
	fit_prop = 0.2
#	mutation_rate = 0.105
	pop_mutation_rate = 0.8
	#Make a history set at some point

	GOAT=[]
	def init_pop(self):
		initdict = {}
		for t in self.types-{0,-1,-2}:
                	initdict[t] = random.randint(1,self.numtags-1)

	        def create_rando(tempdict):
			randict = tempdict.copy()
			for t in random.sample(self.types-{0,-1,-2},2):
				randict[t] = random.randint(0,self.numtags-1)
			randict[-1] = -1
			randict[-2] = -2
			randict[0] = 0
			return([randict,self.f(self.text,randict)])

		self.members = [create_rando(initdict) for _ in range(self.gen_size)]

		
		#sort original members
		self.members.sort(key = lambda x: x[1], reverse = self.maximize)
		

##########modify to take in min or max as goal
        def __init__(self,text,f, gen_size,maximize,numtags):
		self.gen_size = gen_size
		self.text = text
		self.maximize = maximize
		self.types = set([tok for sent in self.text for tok in sent])
		self.numtags = numtags
		self.f = f
                self.init_pop()
		self.GOAT = self.members[:5]
#		self.numtags = 10
#		self.types = sorted(list(set([tok for sent in text for tok in sent])))
	def breed(self,parent1_desc,parent2_desc):
		childict = {}
		for typekey in parent1_desc:
			childict[typekey] = random.choice([parent1_desc[typekey],parent2_desc[typekey]]) 
		return([childict,self.f(self.text,childict)])	

	def mutate(self,indiv):
#		for typekey in self.types-{-1,-2}:# if typekey != 0 and typekey!=1:
#			if self.mutation_rate > random.random():
#				indiv[typekey] = random.randint(1,self.numtags)
		indiv[random.sample(self.types-{0,-1,-2},1)[0]] = random.randint(0,self.numtags-1)
		return([indiv,self.f(self.text,indiv)])
	def next_gen(self):
		self.members.sort(key = lambda x:x[1],reverse = self.maximize) 
                for i in range(self.gen_size):
                        #######Edit this function to account for maximum and minimum
			if self.maximize == True:
                        	if self.members[i][1] > self.GOAT[4][1]:
                                	self.GOAT[4] = self.members[i]
                                	self.GOAT.sort(key = lambda x: x[1], reverse = self.maximize)
				else:
					break
                        elif self.maximize == False:
				if self.GOAT[4][1] > self.members[i][1]:
					self.GOAT[4] = self.members[i]
					self.GOAT.sort(key = lambda x: x[1], reverse = self.maximize)
                   		else:
					break
			else:
				print 'grave error'

		###make sure to include way to indicate either min or max
		fit_num = int(self.fit_prop*self.gen_size)

		parents = self.members[:fit_num]#+ random.sample(self.members[fit_num:],random.randint(0,len(self.members[fit_num:])))
		children =[]
		for i in range(self.gen_size - len(parents)):
			parent1,parent2 = random.sample(self.members,2)
			children.append(self.breed(parent1[0],parent2[0]))

                for i in range(len(parents)):
                        if self.pop_mutation_rate > random.random():
                                parents[i] = self.mutate(parents[i][0])
		self.members = parents+ children


	def get_lineage(self, n):
		for _ in range(n):
			print 'best entropy at generation '+str(_) , self.GOAT[0][1]
			self.next_gen()
		return(self.GOAT, self.members )	


with open('toydata.txt','r') as txtf:
        src_txt = txtf.read()

gettags = sorted(list(set(src_txt.split())-{'the','<s>','</s>'}))
src_txt = [x.split() for x in src_txt.split('\n')]
src_txt = [x for x in src_txt if x!=[]]
forward_key = {}
for i in range(len(gettags)):
	forward_key[gettags[i]]=i+1
forward_key['<s>'] = -1
forward_key['</s>'] = -2
forward_key['the'] = 0
#print forward_key
backward_key = {}
for i in range(len(gettags)):
        backward_key[i+1] = gettags[i]
backward_key[-1] = '<s>'
backward_key[-2] = '</s>'
backward_key[0] = 'the'
#print backward_key

for i in range(len(src_txt)):
	for j in range(len(src_txt[i])):
		src_txt[i][j] = forward_key[src_txt[i][j]]


 

#src_txt = src_txt[:50]
print(len(src_txt))	

start = time.time()
maxi = True
ntags = 5
pop1 = population(src_txt,score,30,maxi,ntags)
Goats,Members = pop1.get_lineage(300)
print 'total time ', time.time()-start


boat = Goats[0][0]
labels = set([boat[x] for x in boat])
eq_classes = {}
w2t = {}
for l in labels:
	eq_classes[l] = [x for x in boat if boat[x] == l]
for x in eq_classes:
	print x, eq_classes[x]

for l in eq_classes:
	for word in eq_classes[l]:
		w2t[word] = l
for x in w2t:
	print x, w2t[x]


n_vocab = len(w2t)
n_docs = len(src_txt)

#transition matrix from i to j
TRM = np.zeros((ntags+2,ntags+2))
for sent in src_txt:
	tagsent = [w2t[tok] for tok in [-1]+sent+[-2]]
	bigrams= zip(tagsent,tagsent[1:])
	for i,j in bigrams:
		TRM[i+1,j+1]+=1
		
for i in range(TRM.shape[0]):
	TRM[i] /=TRM[i].sum()
		
np.save('TRM.npy',TRM)

S2O = np.zeros((ntags+2,n_vocab))
for sent in src_txt:
	pairs = zip([w2t[tok] for tok in sent],sent)
	for i,j in pairs:
		S2O[i+2,j] +=1
	
S2O[0][0] = 1
for i in range(ntags+2):
	S2O[i] /= S2O[i].sum()
np.save('S2O.npy',S2O)









tf_mat = np.zeros(shape = (n_docs,n_vocab))


for i in range(len(src_txt)):
        for j in range(len(src_txt[i])):
                tf_mat[i][src_txt[i][j]]+=1

#print tf_mat
#for x in voc_dict:
#       print x, voc_dict[x]

n_top = 3
lda = skd.LatentDirichletAllocation(n_top,learning_method = 'batch')

#print 'terms:', n_terms
#print 'docs: ',n_docs
#print 'topics: ',n_top


lda.fit(tf_mat)
cPickle.dump(lda,open('ldamodel.pickle','w'))


backward_key = {u+2:v for u,v in backward_key.items()}

cPickle.dump(backward_key,open('key.pickle','w'))


#print lda.components_
#print lda.components_.shape






