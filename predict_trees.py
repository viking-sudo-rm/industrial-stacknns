import torch
from allennlp.data.vocabulary import Vocabulary

from data_readers.brown import BrownDatasetReader
from predictor import TreePredictor
from stack_rnn_LM import StackRNNLanguageModel

def brown_predict(sentence):
    vocab = Vocabulary.from_files("saved_models/vocabulary-brown")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16)

    with open("saved_models/stack-brown.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    dataset_reader = BrownDatasetReader()
    predictor = TreePredictor(model, dataset_reader)
    return predictor.predict(sentence)


def main():
    prediction = brown_predict("AT NP NN JJ")
    print("Prediction", prediction)


if __name__ == "__main__":
    main()
