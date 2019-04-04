import torch
from allennlp.data.vocabulary import Vocabulary

from data_readers.brown import BrownDatasetReader
from stack_rnn_LM import StackRNNLanguageModel


def main():
    vocab = Vocabulary.from_files("saved_models/vocabulary-brown")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16)

    with open("saved_models/stack-brown.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    reader = BrownDatasetReader()
    dataset = reader.read("data/brown.txt")

    for instance in dataset:
        results = model(instance)
        print(results)
        break


if __name__ == "__main__":
    main()