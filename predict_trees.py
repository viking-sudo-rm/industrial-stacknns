import sys

import torch
from allennlp.data.vocabulary import Vocabulary

from build_trees import greedy_parse
from data_readers.brown import BrownDatasetReader
from predictor import TreePredictor
from stack_rnn_LM import StackRNNLanguageModel


def predict_tree(model, sentence, key="pop_strengths"):
    dataset_reader = BrownDatasetReader(labels=False)
    predictor = TreePredictor(model, dataset_reader)
    prediction = predictor.predict(sentence)
    pop_strengths = prediction[key]
    pairs = list(zip(sentence.split(" "), pop_strengths))
    return greedy_parse(pairs)


def main(sentences):
    dataset_name = "linzen"
    swap = True

    vocab = Vocabulary.from_files("saved_models/vocabulary-%s" % dataset_name)
    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  # num_embeddings=10000,
                                  swap_push_pop=True)
    suffix = "-swap" if swap else ""
    with open("saved_models/stack-%s%s.th" % (dataset_name, suffix), "rb") as fh:
        model.load_state_dict(torch.load(fh))

    key = "push_strengths" if swap else "pop_strengths"
    for sentence in sentences:
        sentence = sentence.strip()
        tree = predict_tree(model, sentence, key=key)
        print(tree.to_evalb())


if __name__ == "__main__":
    main(sys.stdin.readlines())
