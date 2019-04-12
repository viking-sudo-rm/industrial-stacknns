from typing import List, Text, Tuple, Union


class InternalBinaryNode:

  def __init__(self, left_child, right_child):
    self.left_tree = left_child
    self.right_tree = right_child

  @staticmethod
  def _child_to_str(child):
    child_str = str(child)
    if isinstance(child, str):
      child_str = "[%s]" % child_str
    return child_str

  def __str__(self):
    """Export the tree in LaTeX forest format."""
    return "[%s%s]" % (self._child_to_str(self.left_tree),
                       self._child_to_str(self.right_tree))

BinaryTree = Union[InternalBinaryNode, Text]


def greedy_parse(scored_tokens: List[Tuple[Text, float]]) -> BinaryTree:
  """This function greedily splits a tree by splitting at the greatest distances.

  To be more precise, this sentence splits recursively at the greatest distance and builds a constituent:

    ((x_<i), (x_i, (x_>i))).

  Args:
    tokens: A list of the words in the sentence.
    distances: A co-indexed list of the syntactic distance scores.

  Returns:
    A tree representing the greedy parse for the sentence.
  """

  # Base case: we have gotten down to one word.
  if len(scored_tokens) == 1:
    return scored_tokens[0][0]

  # Otherwise, we split at the index with greatest distance.
  max_idx, _ = max(enumerate(scored_tokens),
                   key=lambda idx_and_tup: idx_and_tup[1][1])

  specifier = greedy_parse(scored_tokens[:max_idx]) \
      if max_idx != 0 else None
  head = scored_tokens[max_idx][0]
  complement = greedy_parse(scored_tokens[max_idx + 1:]) \
      if max_idx != len(scored_tokens) - 1 else None

  if specifier is None:
    return InternalBinaryNode(head, complement)
  elif complement is None:
    return InternalBinaryNode(specifier, head)
  else:
    return InternalBinaryNode(specifier,
                              InternalBinaryNode(head, complement))
