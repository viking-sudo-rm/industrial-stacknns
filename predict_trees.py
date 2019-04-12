import torch
from allennlp.data.vocabulary import Vocabulary

from build_trees import greedy_parse
from data_readers.brown import BrownDatasetReader
from predictor import TreePredictor
from stack_rnn_LM import StackRNNLanguageModel

def predict(model, sentence):
    dataset_reader = BrownDatasetReader(labels=False)
    predictor = TreePredictor(model, dataset_reader)
    prediction = predictor.predict(sentence)
    pop_strengths = prediction["pop_strengths"]
    pairs = list(zip(sentence.split(" "), pop_strengths))
    return greedy_parse(pairs)


def main():
    # sentence = "AT NN IN AT NNS VBD AT JJ NN"
    sentence = "John and Jill told Mary that she was cool"

    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16)
    with open("saved_models/stack-linzen.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    print(predict(model, sentence))


if __name__ == "__main__":
    main()
