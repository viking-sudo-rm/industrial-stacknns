from predict_trees import predict_tree
from allennlp.data.vocabulary import Vocabulary
from stack_rnn_LM import StackRNNLanguageModel
import torch
import os
import re

from nltk.corpus import BracketParseCorpusReader, treebank
from PYEVALB.scorer import Scorer

pattern = r"\s+"
_PUNCTUATION = {".", ",", ";", "``", "--", "''", ":", "-", "(", ")"}


def clean_nones(t, ignore_periods=False):
    for ind, leaf in reversed(list(enumerate(t.leaves()))):
        postn = t.leaf_treeposition(ind)
        parentpos = postn[:-1]
        if leaf.startswith("*") or \
           t[parentpos].label() == u'-NONE-' or \
           (ignore_periods and leaf == u"."):
            while parentpos and len(t[parentpos]) == 1:
                postn = parentpos
                parentpos = postn[:-1]
            del t[postn]


def gen_words(tree):
    for pos, leaf in enumerate(tree.leaves()):
        parent_pos = tree.leaf_treeposition(pos)[:-1]
        parent = tree[parent_pos]
        if leaf in _PUNCTUATION or parent.label() in _PUNCTUATION:
            continue
        yield leaf


def make_gold_and_test_trees(corpus,
                             path,
                             max_len=None,
                             key="push_strengths",
                             swap=True):
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  # num_embeddings=10000,
                                  swap_push_pop=swap)
    model_name = "linzen-swap" if swap else "linzen"
    with open("saved_models/stack-%s.th" % model_name, "rb") as fh:
        model.load_state_dict(torch.load(fh))

    gold_parses = open(os.path.join(path, "parses.gld"), "w")
    our_parses = open(os.path.join(path, "parses.tst"), "w")

    for ix, parsed_sent in enumerate(corpus.parsed_sents()):
        clean_nones(parsed_sent, ignore_periods=True)

        if max_len is not None and sum(1 for word in
                                       gen_words(parsed_sent)) > max_len:
            continue

        #parsed_sent.chomsky_normal_form()
        parsed_sent.collapse_unary(collapsePOS=True)
        for subtree in parsed_sent.subtrees():
            subtree.set_label("X")
        start_pos = parsed_sent.leaf_treeposition(0)
        parsed_sent[start_pos] = parsed_sent[start_pos].lower()

        # if max_len is not None and sum(1 for c in parsed_sent.flatten()
        #                                if c not in _PUNCTUATION) > max_len:
        #     continue

        gold_oneline_parse = re.sub(pattern, " ", str(parsed_sent))
        gold_parses.write(gold_oneline_parse + "\n")

        tokens = " ".join(tok for tok in parsed_sent.flatten())
        our_tree = predict_tree(model, tokens, key=key)
        try:
            our_oneline_parse = our_tree.to_evalb()
        except Exception:
            # In the one-word cases (very few), we just assign the standard
            # parse to avoid throwing errors.
            print(our_tree)
            our_oneline_parse = "(X (X %s))" % our_tree
        our_parses.write(our_oneline_parse + "\n")

        if ix == 0:
            print(gold_oneline_parse)
            print(our_oneline_parse)

    gold_parses.close()
    our_parses.close()

def score_trees(path):
    scorer = Scorer()
    gold_path = os.path.join(path, "parses.gld")
    test_path = os.path.join(path, "parses.tst")
    results_path = os.path.join(path, "results.txt")
    scorer.evalb(gold_path, test_path, results_path)

if __name__ == "__main__":
    # The part of the corpus included in NLTK.
    # make_gold_and_test_trees(treebank, "predictions/wsj-nltk")
    # path = "predictions/wsj-nltk"

    # The standard section for evaluation: WSJ23.
    corpus_root = "data/treebank_3/parsed/mrg/wsj/23"
    corpus = BracketParseCorpusReader(corpus_root, r".*\.mrg")
    print("Files:", corpus.fileids())
    path = "predictions/wsj-23-naive"
    make_gold_and_test_trees(corpus, path, key="pop_strengths")

    # The whole corpus with length <= 10.
    # corpus_root = "data/treebank_3/parsed/mrg/wsj"
    # corpus = BracketParseCorpusReader(corpus_root, r".*\.mrg")
    # path = "predictions/wsj-10-naive"
    # make_gold_and_test_trees(corpus, path, max_len=10, key="pop_strengths")

    # Do the actual scoring.
    score_trees(path)
