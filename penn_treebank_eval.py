from predict_trees import predict_tree
from allennlp.data.vocabulary import Vocabulary
from stack_rnn_LM import StackRNNLanguageModel
import torch
import json
import numpy as np

from nltk.tree import Tree
# from nltk.corpus import BracketParseCorpusReader
# from PYEVALB.scorer import Scorer

from build_trees import InternalBinaryNode

_PUNCTUATION = {".", ",", ";", "``", "--", "''", ":", "-", "(", ")"}


def gen_sentence(tree):
    """Return all leaf tokens in correct order."""
    if isinstance(tree, str):
        yield tree
    else:
        for child in tree:
            yield from gen_sentence(child)


def find_and_decapitalize_first_word(tree):
    if isinstance(tree, str):
        return True
    else:
        if find_and_decapitalize_first_word(tree[0]):
            tree[0] = tree[0].lower()
        return False


def get_tree_without_periods(tree):
    if isinstance(tree, str):
        return None if tree == "." else tree

    left_child = get_tree_without_periods(tree[0])
    right_child = get_tree_without_periods(tree[1])

    if left_child is None:
        return right_child
    if right_child is None:
        return left_child
    else:
        return [left_child, right_child]


def gen_tree_pairs(model,
                   gold_parses,
                   max_len=None,
                   key="push_strengths",
                   swap=True,
                   decapitalize_first_word=True,
                   mock_right_branching=False,
                   remove_periods=False):
    for ix, gold_parse in enumerate(gold_parses):

        if decapitalize_first_word:
            find_and_decapitalize_first_word(gold_parse)

        if remove_periods:
            gold_parse = get_tree_without_periods(gold_parse)

        token_generator = gen_sentence(gold_parse)
        if not mock_right_branching:
            our_parse = predict_tree(model, " ".join(token_generator), key=key)
            our_parse = InternalBinaryNode.to_nested_lists(our_parse)
        else:
            tokens = list(token_generator)
            our_parse = right_branching_parse(tokens)

        yield gold_parse, our_parse


def get_brackets(tree, idx=0):
    """Taken from
    https://github.com/yikangshen/PRPN/blob/master/test_phrase_grammar.py"""
    # TODO: Compare to to_indexed_contituents in Htut et al.
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

    # Note: for binary trees, all three metrics should be the same.
    return precision, recall, f1


def tree_to_nested_lists(tree):
    """Convert an NLTK tree to an unlabelled nested list format."""
    children = [child for child in tree]
    if len(children) == 1 and isinstance(children[0], str):
        return children[0]
    else:
        return [tree_to_nested_lists(child) for child in children]


def eval_trees(tree_pairs):
    """Take a generator of (gold, predicted) trees and evaluate performance."""
    metrics = []
    for gold_tree, our_tree in tree_pairs:
        print("=" * 50)
        print("--- Gold Parse ---")
        print(gold_tree)
        print("--- Our Parse ---")
        print(our_tree)
        metrics.append(get_p_r_f1(gold_tree, our_tree))

    print("\n\n" + "=" * 50)
    precisions, recalls, f1s = zip(*metrics)
    print("Pr:", np.mean(precisions))
    print("Re:", np.mean(recalls))
    print("F1:", np.mean(f1s))


def right_branching_parse(tokens):
    if len(tokens) == 1:
        return tokens[0]
    else:
        return [tokens[0], right_branching_parse(tokens[1:])]


def htut_parse_to_nested_lists(htut_string):
    tokens = htut_string.split(" ")
    stack = []
    for token in tokens:
        if token == "(":
            continue
        elif token == ")":  # Reduce.
            right_child = stack.pop(0)
            left_child = stack.pop(0)
            stack.insert(0, [left_child, right_child])
        else:  # Shift.
            stack.insert(0, token)
    return stack[0]


def gen_htut_nested_lists(filename):
    """Read Htut-formatted binary trees from a .jsonl file."""
    with open(filename) as in_file:
        for line in in_file:
            json_dict = json.loads(line)
            htut_parse = json_dict["sentence1_binary_parse"]
            yield htut_parse_to_nested_lists(htut_parse)


if __name__ == "__main__":
    # Specify, model, architectural parameters, and parsing options.
    model_name = "linzen"
    swap = True
    kwargs = {
        "mock_right_branching": False,
        "decapitalize_first_word": True,
        "remove_periods": False,
    }

    # Paths and parsing options depend on whether we use push or pop strength.
    vocab_path = "saved_models/vocabulary-%s" % model_name
    if swap:
        model_path = "saved_models/stack-%s-swap.th" % model_name
        kwargs["key"] = "push_strengths"
    else:
        model_path = "saved_models/stack-%s.th" % model_name
        kwargs["key"] = "pop_strength"

    # Load the vocabulary and model.
    vocab = Vocabulary.from_files(vocab_path)
    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  # num_embeddings=10000,
                                  swap_push_pop=swap)
    with open(model_path, "rb") as fh:
        model.load_state_dict(torch.load(fh))

    # Htut et al.'s binarized version of the WSJ23 corpus.
    trees = gen_htut_nested_lists("data/ptb_sec23.jsonl")

    # Generate and evaluate predicted trees.
    tree_pairs = gen_tree_pairs(model, trees, **kwargs)
    eval_trees(tree_pairs)
