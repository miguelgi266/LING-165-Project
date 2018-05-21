import nltk, os, string, cPickle


pos_tagger = cPickle.load(open('pos_tagger.pickle','rb'))

def get_blueprint(fpath):
	with open(fpath,'r') as f:
		poem = f.read()
		poemc = poem[:]
		if '\'s' in poem or '\'nt' in poem:
			return None
		lengths = [len([tok for tok in stanza.split() if tok !='``']) for stanza in poem.split('\n') if len(stanza)>=1]
		poem = nltk.sent_tokenize(poem)
		poem = [nltk.word_tokenize(sent) for sent in poem]
		print poem
#		poem = [nltk.pos_tag(sent) for sent in poem]
#		poem = [nltk.NgramTagger(n=2,model = None).tag(sent) for sent in poem]
		poem = [pos_tagger.tag(sent) for sent in poem]
#		print [pair for sent in poem for pair in sent]
		poem = [pair for sent in poem for pair in sent if pair[0]!= '``' and pair[0] not in string.punctuation and pair[1] not in string.punctuation]

		print poem
		tags = [pair[1] for pair in poem]
		return {'tags':tags,'lengths':lengths}

hnames =  os.listdir('haikus')
templates = []
for flnm in hnames:
	struct =  get_blueprint('haikus/'+flnm)
	if struct != None:
		templates.append(struct)
#		print struct
		if len(templates[-1]['lengths'])!=3:
			print len(templates[-1]['lengths'])

cPickle.dump(templates, open('templates.pickle','w'))
