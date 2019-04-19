import argparse
import logging

import torch
from allennlp.data.vocabulary import Vocabulary

from build_trees import greedy_parse, InternalBinaryNode
from data_readers.brown import BrownDatasetReader
from predictor import TreePredictor
from stack_rnn_LM import StackRNNLanguageModel


TREE_ENCODINGS = {
    "evalb": InternalBinaryNode.to_evalb,
    "latex": InternalBinaryNode.to_latex,
}


def predict_tree(model, sentence, key="pop_strengths"):
    dataset_reader = BrownDatasetReader(labels=False)
    predictor = TreePredictor(model, dataset_reader)
    prediction = predictor.predict(sentence)
    pop_strengths = prediction[key]
    if all(dist == 1 for dist in pop_strengths):
        logging.warning("All syntactic distances are 1. Is there a mismatch?")
    pairs = list(zip(sentence.split(" "), pop_strengths))
    return greedy_parse(pairs)


def main(sentence, dataset_name, swap, num_embeddings, encoding_fn):
    """Parse an arbitrary sentence with a pretrained model.

    Example usage from the command line:
    python3 predict_trees.py "AT NNS VBD AT JJ JJ NN" --dataset brown --no_swap
    """
    vocab = Vocabulary.from_files("saved_models/vocabulary-%s" % dataset_name)
    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  num_embeddings=num_embeddings,
                                  swap_push_pop=swap)
    dataset_name += "-swap" if swap else ""
    with open("saved_models/stack-%s.th" % dataset_name, "rb") as fh:
        model.load_state_dict(torch.load(fh))

    key = "push_strengths" if swap else "pop_strengths"
    tree = predict_tree(model, sentence, key=key)
    print(encoding_fn(tree))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str)
    parser.add_argument("--dataset",
                        choices=["brown", "linzen"],
                        default="linzen")
    parser.add_argument("--num_embeddings", type=int, default=None)
    parser.add_argument("--no_swap", action="store_true")
    parser.add_argument("--enc",
                        choices=["evalb", "latex"],
                        default="latex")
    args = parser.parse_args()

    main(args.sentence,
         args.dataset,
         not args.no_swap,
         args.num_embeddings,
         TREE_ENCODINGS[args.enc])
