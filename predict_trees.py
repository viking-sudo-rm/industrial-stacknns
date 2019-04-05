import torch
from allennlp.data.vocabulary import Vocabulary

from build_trees import greedy_parse
from data_readers.brown import BrownDatasetReader
from predictor import TreePredictor
from stack_rnn_LM import StackRNNLanguageModel

def brown_predict(sentence):
    vocab = Vocabulary.from_files("saved_models/vocabulary-brown")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16)

    with open("saved_models/stack-brown.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    dataset_reader = BrownDatasetReader(labels=False)
    predictor = TreePredictor(model, dataset_reader)
    return predictor.predict(sentence)


def main():
    sentence = "AT NN IN AT NNS VBD AT JJ NN"
    prediction = brown_predict(sentence)
    pop_strengths = prediction["pop_strengths"]

    print("Parsing..")
    pairs = list(zip(sentence.split(" "), pop_strengths))
    parse = greedy_parse(pairs)
    print("Got parse", parse)


if __name__ == "__main__":
    main()
