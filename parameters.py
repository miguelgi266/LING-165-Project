import nltk, cPickle
import numpy as np
pos_tagger = cPickle.load(open('pos_tagger.pickle','rb'))
def cos_msr(a,b):
	return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

d = nltk.corpus.cmudict.dict()
#code for makeshift and nysl taken from 
def makeshift(word):
        count = 0.0
        vowels = 'aeiouy'
        word = word.lower()
        if word[0] in vowels:
                count +=1
        for index in range(1,len(word)):
                if word[index] in vowels and word[index-1] not in vowels:
                        count +=1
        if word.endswith('e'):
                count -=1
        if word.endswith('le'):
                count+=1
        if count == 0:
                count +=1
        return count


def nsyl(word):
        try:
                syll_count = 0.0
                for phone in d[word.lower()][0]:
                        if phone[-1].isdigit():
                                syll_count+=1
                return syll_count

        except KeyError:
                return makeshift(word)


def open_proc(flnm):
	with open(flnm,'r') as f:
		txt = f.read().decode('utf8')
	sents = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(txt)]
	origsents = sents[:]
	sents = [[tok.lower() for tok in sent] for sent in sents]
	tgsnt =[]
	for i in range(len(origsents)):
                tgsnt+=[(word.lower(),tag) for word,tag in nltk.pos_tag(origsents[i])]
	words = sorted(list(set([pair[0] for pair in tgsnt])))
	tags = sorted(list(set([pair[1] for pair in tgsnt])))
	w2n = {words[i]:i for i in range(len(words))} 
	t2n = {tags[i]:i for i in range(len(tags))}
	tagpairs =[(w2n[pair[0]],t2n[pair[1]]) for pair in tgsnt]
	sents = [[w2n[tok] for tok in sent] for sent in sents]
	return origsents,sents,tagpairs, words,tags,w2n,t2n

def parameter_matrices(tagpairs,sents,words,n_terms,n_tags,n2t):
	pos = np.zeros((n_terms,n_tags))
	s = np.zeros(n_terms)
	F = np.zeros((n_terms,n_terms))

	for i,j in tagpairs:
#		print words[i],n2t[j]
        	pos[i,j]=1

#	for j in range(n_tags):
#		print n2t[j],'.............'
#		for i in range(n_terms):
#			if pos[i,j] == 1:
#				print words[i]



	for i in range(n_terms):
		s[i] = nsyl(words[i])
        np.save('s.npy',s)#,allow_pickle = False
	for sent in sents:
#		print 'sentlen:',len(sent)
        	for i in range(len(sent)):
#			print i+1,i+5
                	for w_idx in sent[i+1:i+4]:
                        	F[sent[i],w_idx]+=1
#	F = (F==0).astype('float32')
	u,s,v = np.linalg.svd(F[:],full_matrices = False)
	c = min(len(s)/3,100)
	q2k = u[:,:c].dot(np.diag(s[:c]))
	np.save('q2k.npy',q2k)
	np.save('pos.npy',pos)
	np.save('T.npy',F)
#	print 'vocab size %d, tag size %d' %pos.shape


def best_k(origsents,sents,doc,w2n):
	M = np.zeros((len(words),len(sents)))
#	print M.shape
	for j in range(len(sents)):
		for w in sents[j]:
			M[w,j]+=1
	idf = np.zeros(M.shape[0])

	#determine idf for each term and modify tf matrix accordingly
	#by end of loop tf matrix should be a tf-idf matrix
	for i in range(M.shape[0]):
        	idf[i] = np.log(float(M.shape[1])/((M[i]!=0.0).sum()))
        	M[i,:] = idf[i]*M[i,:]
	U,S,VT = np.linalg.svd(M, full_matrices = False)

	k = 500
	S = np.diag(S[:k])
	U = U[:,:k]
	VT = VT[:k,:]

	D_k = S.dot(VT) #S_k * VT_k; reduced document representation
	full2k = np.diag(1.0/np.diag(S)).dot(U.T) #S_k^{-1} * VT_k #maps query vector to R^{k}

	q = np.zeros(len(w2n))
	for word in doc:
		q[w2n[word]]+=idf[w2n[word]]
	q = full2k.dot(q)

	scores = [(i, cos_msr(q,D_k[:,i])) for i in range(D_k.shape[1])]
	scores.sort(key = lambda x:x[1],reverse = True)
	scores = [x[0] for x in scores]
	n2w = {v:u for u,v in w2n.items()}
	
	vl = []
	i = 0
	for idx in scores:
		i += 1
		vl += sents[idx]
		if len(set(vl)) >= 200:
			break

	sents =   [ origsents[idx] for idx in scores[:i]]
	return sents

def reduced_rep(sents):
        tgsnt =[]
        for sent in sents:
		temp = [pair[1] for pair in pos_tagger.tag(sent)]
#		print temp
                tgsnt.append(zip([tok.lower() for tok in sent],temp))
#		print tgsnt[-1]
	sents = [[tok.lower() for tok in sent] for sent in sents]
        words = sorted(list(set([pair[0] for sent in tgsnt for pair in sent])))
        tags = sorted(list(set([pair[1] for sent in tgsnt for pair in sent])))
        w2n = {words[i]:i for i in range(len(words))}
#	print w2n.keys()
        t2n = {tags[i]:i for i in range(len(tags))}
#	print t2n.keys()
        tagpairs =[(w2n[pair[0]],t2n[pair[1]]) for sent in tgsnt for pair in sent]
        sents = [[w2n[tok] for tok in sent] for sent in sents]
        return sents,tagpairs, words,tags,w2n,t2n







doc =  ['mountain']
#doc = ['mother','picture']
#doc = ['shoes']

flnm = 'texts/outings.txt'
origsents,sents,tagpairs,words,tags,w2n,t2n = open_proc(flnm)
sents = best_k(origsents,sents,doc,w2n)
del tagpairs,words,tags,w2n,t2n 
sents, tagpairs, words, tags, w2n, t2n =  reduced_rep(sents)


n_tags = len(tags)
n_terms = len(words)
with open('w2n.pickle','wb') as f:
	cPickle.dump(w2n,f)
with open('t2n.pickle','wb') as f:
	cPickle.dump(t2n,f)

parameter_matrices(tagpairs,sents,words,n_terms,n_tags,{v:u for u,v in t2n.items()})




