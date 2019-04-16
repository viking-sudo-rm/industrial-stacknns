import os
import sys
from predict_trees import predict_tree
from allennlp.data.vocabulary import Vocabulary
from stack_rnn_LM import StackRNNLanguageModel
import torch
import re

from nltk.corpus import treebank
from PYEVALB.scorer import Scorer
from PYEVALB import parser

pattern = r"\s+"


def clean_nones(t): #nltk_tree
    for ind, leaf in reversed(list(enumerate(t.leaves()))):
        postn = t.leaf_treeposition(ind)
        parentpos = postn[:-1]
        if leaf.startswith("*") or t[parentpos].label() == u'-NONE-': #and leaf.endswith("*"):
            while parentpos and len(t[parentpos]) == 1:
                postn = parentpos
                parentpos = postn[:-1]
            # print(t[postn], "will be deleted")
            del t[postn]

def make_gold_and_test_trees():
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  # num_embeddings=10000,
                                  swap_push_pop=True)
    with open("saved_models/stack-linzen-swap.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    gold_parses = open("predictions/gold_parses.gld", "w")
    our_parses = open("predictions/our_parses.tst", "w")

    for ix, parsed_sent in enumerate(treebank.parsed_sents()):
        clean_nones(parsed_sent)
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
            our_oneline_parse = "(X (X %s))" % our_tree
        our_parses.write(our_oneline_parse+"\n")

        if ix==0:
            print(gold_oneline_parse)
            print(our_oneline_parse)

    gold_parses.close()
    our_parses.close()


def score_trees():
    scorer = Scorer()
    gold_path = "predictions/wsj-nltk/parses.gld"
    test_path = "predictions/wsj-nltk/parses.tst"
    results_path = "predictions/wsj-nltk/results.txt"
    scorer.evalb(gold_path, test_path, results_path)

    # gold = "(X (X (X (X Pierre) (X Vinken)) (X (X ,) (X (X (X (X 61) (X years)) (X old)) (X ,)))) (X (X (X will) (X (X join) (X (X (X the) (X board)) (X (X (X as) (X (X a) (X (X nonexecutive) (X director)))) (X (X Nov.) (X 29)))))) (X .)))"
    # test = "(X (X (X (X (X Pierre) (X Vinken)) (X (X ,) (X (X (X (X 61) (X (X years) (X old))) (X ,)) (X will)))) (X (X join) (X (X the) (X board)))) (X (X as) (X (X (X a) (X (X nonexecutive) (X director))) (X (X Nov.) (X (X 29) (X .))))))"
    
    # with open("predictions/gold_parses.gld") as gold_file:
    #     gold_lines = gold_file.readlines()
    # with open("predictions/our_parses.tst") as our_file:
    #     our_lines = our_file.readlines()

    # for gold_line, our_line in zip(gold_lines, our_lines):
    #     print("=" * 50)
    #     print(gold_line)
    #     print(our_line)
    #     gold_tree = parser.create_from_bracket_string(gold_line)
    #     our_tree = parser.create_from_bracket_string(our_line)
    #     result = scorer.score_trees(gold_tree, our_tree)


if __name__ == "__main__":
    # make_gold_and_test_trees()
    score_trees()
