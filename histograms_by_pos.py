import matplotlib.pyplot as plt
import numpy as np

import torch
from predictor import TreePredictor
from data_readers.brown import BrownDatasetReader
from data_readers.linzen import LinzenLMDatasetReader
from allennlp.data.vocabulary import Vocabulary
from stack_rnn_LM import StackRNNLanguageModel

def predictions(model, sentence):
	dataset_reader = BrownDatasetReader(labels=False)
	predictor = TreePredictor(model, dataset_reader)
	return predictor.predict(sentence)

def makeModel(dataset_name):
	swap = True #keeping pop constant

	vocab = Vocabulary.from_files("saved_models/vocabulary-%s" % dataset_name)
	model = StackRNNLanguageModel(vocab,
	                          rnn_dim=100,
	                          stack_dim=16,
	                          num_embeddings=10000,
	                          swap_push_pop=True)
	if (dataset_name == "linzen"):
		model = StackRNNLanguageModel(vocab,
	                          rnn_dim=100,
	                          stack_dim=16,
	                          #num_embeddings=10000,
	                          swap_push_pop=True)

	suffix = "-swap" if swap else ""
	with open("saved_models/stack-%s%s.th" % (dataset_name, suffix), "rb") as fh:
		model.load_state_dict(torch.load(fh))

	dataset_reader = BrownDatasetReader(labels=False)
	sentences = dataset_reader._read("brown.txt")
	if (dataset_name == "linzen"):
		dataset_reader = LinzenLMDatasetReader()
		sentences = dataset_reader._read("linzen_lm_data/agr_50_mostcommon_10K.tsv.90")

	return (model, sentences)

def makeHistograms(dataset, part):
	model, sentences = makeModel(dataset)
	allEvents = []
	DTATEvent = []
	NNEvent = []
	VBDEvent = []
	INEvent = []

	i = 0
	for instance in sentences:
		i += 1
		if (i % 10 == 0):
			print("Progress: ", i/100)
		if (i > 100):
			break
		sentence = ""
		for word in instance["sentence"]:
			sentence += str(word) + " "
		res = predictions(model, sentence)
		for index, p in enumerate(res['%s_strengths' % part]):
			if (p > 0.635 and p < 0.65):
				print(index, sentence)

			allEvents.append(p)
			if index < len(instance["sentence"]):
				#print(index, str(instance["pos"][index]))
				if str(instance["sentence"][index]) == "AT" or (dataset == "linzen" and instance["pos"][index] == "DT"):
					DTATEvent.append(p)
				if str(instance["sentence"][index]) == "NN" or (dataset == "linzen" and instance["pos"][index] == "NN" or instance["pos"][index] == "NNS"):
					NNEvent.append(p)
				if str(instance["sentence"][index]) == "VBD" or (dataset == "linzen" and instance["pos"][index] in ["VBD", "VBZ", "VBG"]):
					VBDEvent.append(p)
				if (dataset == "linzen" and instance["pos"][index] == "IN"):
					INEvent.append(p)

	print("all push mean: ", np.mean(allEvents))
	#print("AT mean: ", np.mean(DTATEvent))
	#plt.hist(allEvents, density=True, bins=100, label = 'All')
	plt.hist([VBDEvent, DTATEvent, INEvent], bins=100, label = ['Verbs', 'AT/DT', 'IN'])
	plt.legend(loc='upper right')
	plt.xlabel('%s strength' % part);
	plt.ylabel('Freq');
	plt.show()

def main():
	makeHistograms("linzen", "push")

if __name__ == "__main__":
	main()


