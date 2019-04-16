import matplotlib.pyplot as plt
import numpy as np

from predict_trees import brown_predict
from data_readers.brown import BrownDatasetReader

def main():
	sent = "AT NN IN AT NNS VBD AT JJ NN"
	results = brown_predict(sent)
	print(sent)
	print("push_strengths", results['push_strengths'])
	print("pop_strengths", results['pop_strengths'])
	print("read_strengths", results['read_strengths'])

	dataset_reader = BrownDatasetReader(labels=False)
	sentences = dataset_reader._read("brown.txt")

	i = 0
	pop = []
	ATpop = []
	NNpop = []
	VBDpop = []
	for instance in sentences:
		i += 1
		if (i % 100 == 0):
			print("Progress: ", i/1000)
		if (i > 1000):
			break
		sentence = ""
		for word in instance["sentence"]:
			sentence += str(word) + " "
		res = brown_predict(sentence)
		for index, p in enumerate(res['pop_strengths']):
			pop.append(p)
			# print(i, str(instance["sentence"][i]))
			if index < len(instance["sentence"]):
				if str(instance["sentence"][index]) == "AT":
					ATpop.append(p)
				if str(instance["sentence"][index]) == "NN":
					NNpop.append(p)
				if str(instance["sentence"][index]) == "VBD":
					VBDpop.append(p)



	#%matplotlib inline
	print("all pop mean: ", np.mean(pop))
	#print("NN mean: ", np.mean(NNpop))
	print("AT mean: ", np.mean(ATpop))
	#print("VBD mean: ", np.mean(VBDpop))
	plt.hist(pop, density=True, bins=100, label = 'All')
	#plt.hist(NNpop, density=True, bins=100, label = 'NN')
	plt.hist(ATpop, density=True, bins=100, label = 'AT')
	#plt.hist(VBDpop, density=True, bins=100, label = 'VBD')
	plt.legend(loc='upper right')
	plt.ylabel('Probability');
	plt.show()

		# print(str(instance["sentence"][0]))

if __name__ == "__main__":
	main()