from itertools import permutations


def ngrams(text, n, permutation=True):
	"""
	Computes ngrams for text
	Args:
		text (str): text whose ngrams has to be find
		n (int): ngram value
		permutation (bool): if True will form ngrams(>1) with all possible arrangements
	Returns:
		output (list) : list of all ngrams formed
	"""
	text = text.split()
	output = []
	for i in range(len(text) - n + 1):
		if permutation:
			for each in permutations(text[i:i + n]):
				output.append(" ".join(each))
		else:
			output.append(" ".join(text[i:i + n]))
	return output
