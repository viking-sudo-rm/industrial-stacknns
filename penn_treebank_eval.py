from predict_trees import predict_tree
from allennlp.data.vocabulary import Vocabulary
from stack_rnn_LM import StackRNNLanguageModel
import torch
import numpy as np

from nltk.tree import Tree
from nltk.corpus import BracketParseCorpusReader
# from PYEVALB.scorer import Scorer

from build_trees import InternalBinaryNode

pattern = r"\s+"
_PUNCTUATION = {".", ",", ";", "``", "--", "''", ":", "-", "(", ")"}


def remove_nulls(t, ignore_periods=False):
    for ind, leaf in reversed(list(enumerate(t.leaves()))):
        postn = t.leaf_treeposition(ind)
        parentpos = postn[:-1]

        # Remove null elements.
        if leaf.startswith("*") or \
           t[parentpos].label() == u'-NONE-' or \
           (ignore_periods and leaf == u"."):
            while parentpos and len(t[parentpos]) == 1:
                postn = parentpos
                parentpos = postn[:-1]
            del t[postn]


def gen_words(tree):
    """Returns words, but not in correct order."""
    for pos, leaf in tree.leaves():
        parent_pos = tree.leaf_treeposition(pos)[:-1]
        parent = tree[parent_pos]
        if leaf in _PUNCTUATION or parent.label() in _PUNCTUATION:
            continue
        yield leaf


def gen_sentence(tree):
    """Return all leaf tokens in correct order."""
    if isinstance(tree, str):
        yield tree
    else:
        for child in tree:
            yield from gen_sentence(child)


def gen_gold_and_test_trees(model,
                            corpus,
                            path,
                            max_len=None,
                            key="push_strengths",
                            swap=True,
                            decapitalize_first_word=True,
                            ignore_periods=True,
                            mock_right_branching=False):
    for ix, gold_parse in enumerate(corpus.parsed_sents()):

        # Ignore long sentences (> max_len).
        if max_len is not None and sum(1 for word in
                                       gen_words(gold_parse)) > max_len:
            continue

        # Here are some modifications that can be made to the trees.
        remove_nulls(gold_parse, ignore_periods=ignore_periods)
        # gold_parse.chomsky_normal_form()
        # gold_parse.collapse_unary(collapsePOS=True)

        if decapitalize_first_word:
            start_pos = gold_parse.leaf_treeposition(0)
            gold_parse[start_pos] = gold_parse[start_pos].lower()

        token_generator = gen_sentence(gold_parse)
        if not mock_right_branching:
            our_parse = predict_tree(model, " ".join(token_generator), key=key)
        else:
            tokens = list(token_generator)
            if len(tokens) > 12:
                continue
            our_parse = right_branching_parse(tokens)

        yield gold_parse, our_parse


def right_branching_parse(tokens):
    if len(tokens) == 1:
        return tokens[0]
    else:
        return [tokens[0], right_branching_parse(tokens[1:])]


def get_brackets(tree, idx=0):
    """Taken from
    https://github.com/yikangshen/PRPN/blob/master/test_phrase_grammar.py"""
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def get_p_r_f1(gold_tree, our_tree):
    """Taken from
    https://github.com/yikangshen/PRPN/blob/master/test_phrase_grammar.py"""
    model_out, _ = get_brackets(our_tree)
    std_out, _ = get_brackets(gold_tree)
    overlap = model_out.intersection(std_out)

    precision = float(len(overlap)) / (len(model_out) + 1e-8)
    recall = float(len(overlap)) / (len(std_out) + 1e-8)
    if len(std_out) == 0:
        recall = 1.
        if len(model_out) == 0:
            precision = 1.
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def tree_to_nested_lists(tree):
    children = [child for child in tree]
    if len(children) == 1 and isinstance(children[0], str):
        return children[0]
    else:
        return [tree_to_nested_lists(child) for child in children]


def eval_trees(tree_pairs):
    """Take a generator of (gold, predicted) trees and evaluate performance."""
    metrics = []

    for gold_tree, our_tree in tree_pairs:
        gold_nested_lists = tree_to_nested_lists(gold_tree)
        our_nested_lists = InternalBinaryNode.to_nested_lists(our_tree)

        print("=" * 50)
        print("--- Gold Parse ---")
        print(gold_nested_lists)
        # gold_tree.pretty_print()
        print("--- Our Parse ---")
        print(our_nested_lists)
        # Tree.fromstring(our_tree.to_evalb()).pretty_print()

        # Nested lists let us avoid issues with POS positions/lack of labels.
        metrics.append(get_p_r_f1(gold_nested_lists, our_nested_lists))

    print("\n\n" + "=" * 50)
    precisions, recalls, f1s = zip(*metrics)
    print("Pr:", np.mean(precisions))
    print("Re:", np.mean(recalls))
    print("F1:", np.mean(f1s))


if __name__ == "__main__":
    # Load the trained Linzen model.
    model_name = "linzen"
    vocab_path = "saved_models/vocabulary-linzen"
    swap = True
    kwargs = {
        "decapitalize_first_word": True,
        "ignore_periods": False,
        "mock_right_branching": True,
    }
    if swap:
        model_name += "-swap"
        kwargs["key"] = "push_strengths"
    else:
        kwargs["key"] = "pop_strength"

    vocab = Vocabulary.from_files(vocab_path)
    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  # num_embeddings=10000,
                                  swap_push_pop=swap)
    with open("saved_models/stack-%s.th" % model_name, "rb") as fh:
        model.load_state_dict(torch.load(fh))

    # The standard section for evaluation: WSJ23.
    corpus_root = "data/treebank_3/parsed/mrg/wsj/23"
    corpus = BracketParseCorpusReader(corpus_root, r".*\.mrg")
    path = "predictions/%s/wsj-23" % model_name
    kwargs["max_len"] = None  # Deprecated.

    tree_pairs = gen_gold_and_test_trees(model,
                                         corpus,
                                         path,
                                         **kwargs)

    eval_trees(tree_pairs)
