import os
import sys
from predict_trees import predict_tree
from allennlp.data.vocabulary import Vocabulary
from stack_rnn_LM import StackRNNLanguageModel
import torch
import re

pattern = r"\s+"

from nltk.corpus import treebank

def clean_nones(t): #nltk_tree
    for ind, leaf in reversed(list(enumerate(t.leaves()))):
        postn = t.leaf_treeposition(ind)
        parentpos = postn[:-1]
        if leaf.startswith("*") or t[parentpos].label() == u'-NONE-': #and leaf.endswith("*"):
            while parentpos and len(t[parentpos]) == 1:
                postn = parentpos
                parentpos = postn[:-1]
            print(t[postn], "will be deleted")
            del t[postn]

def main():
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  # num_embeddings=10000,
                                  swap_push_pop=True)
    with open("saved_models/stack-linzen-swap.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    gold_parses = open("predictions/gold_parses.txt", "w")
    our_parses = open("predictions/our_parses.txt", "w")

    for ix, parsed_sent in enumerate(treebank.parsed_sents()):
        parsed_sent.chomsky_normal_form()
        for subtree in parsed_sent.subtrees():
            subtree.set_label("X")

        gold_oneline_parse = re.sub(pattern, " ", str(parsed_sent))
        gold_parses.write(gold_oneline_parse+"\n")

        tokens = " ".join(tok for tok in parsed_sent.flatten())
        our_tree = predict_tree(model, tokens, key="push_strengths")
        try:
            our_oneline_parse = our_tree.to_evalb()
        except:
            print(our_tree)
            our_oneline_parse = "(X %s)" % our_tree
        our_parses.write(our_oneline_parse+"\n")

        if ix==0:
            print(gold_oneline_parse)
            print(our_oneline_parse)

    gold_parses.close()
    our_parses.close()



if __name__ == "__main__":
    main()
