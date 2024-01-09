import nltk, pickle



regexptagger = nltk.tag.RegexpTagger([ (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),(r'(The|the|A|a|An|an)$', 'AT'),(r'.*able$', 'JJ'),
 (r'.*ness$', 'NN'),(r'.*ly$', 'RB'), (r'.*s$', 'NNS'), (r'.*ing$', 'VBG'), (r'.*ed$', 'VBD'), (r'.*', 'NN')])

 
treebank_cutoff = len(nltk.corpus.treebank.tagged_sents()) * 3 // 4
train_sents = nltk.corpus.treebank.tagged_sents()[:treebank_cutoff]
treebank_test = nltk.corpus.treebank.tagged_sents()[treebank_cutoff:]
#train_sents  = train_sents + treebank_test


tagger = nltk.AffixTagger(train_sents,backoff = regexptagger)
tagger = nltk.UnigramTagger(train_sents, backoff = tagger)
tagger = nltk.BigramTagger(train_sents,backoff = tagger)
tagger = nltk.TrigramTagger(train_sents,backoff = tagger)

pickle.dump(tagger,open('pos_tagger.pickle','wb'))


print(tagger.evaluate(treebank_test))
