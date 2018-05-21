from nltk.corpus import cmudict
d = cmudict.dict()
def makeshift(word):
	count = 0
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
		syll_count = 0
		for phone in d[word.lower()][0]:
			if phone[-1].isdigit():
				syll_count+=1	
		return syll_count

	except KeyError:
		return makeshift(word)


