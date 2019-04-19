from typing import List, Text, Tuple, Union


class InternalBinaryNode:

  def __init__(self, left_child, right_child, label=None):
    self.left_tree = left_child
    self.right_tree = right_child
    self.label = label

  @staticmethod
  def _child_to_latex(child, labels=False):
    if isinstance(child, InternalBinaryNode):
      return child.to_latex(labels=labels)
    else:
      return "%s" % str(child)

  def to_latex(self, labels=False):
    """Export the tree in LaTeX forest format."""
    if labels:
      return "[.%s %s %s ]" % (self.label,
                            self._child_to_latex(self.left_tree, True),
                            self._child_to_latex(self.right_tree, True))
    else:
      return "[ %s %s ]" % (self._child_to_latex(self.left_tree),
                         self._child_to_latex(self.right_tree))

  @staticmethod
  def _child_to_evalb(child):
    if isinstance(child, InternalBinaryNode):
      return child.to_evalb()
    else:
      return "(X %s)" % str(child)

  def to_evalb(self):
    """Export the tree to PARSE EVAL format."""
    return "(X %s %s)" % (self._child_to_evalb(self.left_tree),
                          self._child_to_evalb(self.right_tree))


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
    return InternalBinaryNode(head, complement, label="XP")
  elif complement is None:
    return InternalBinaryNode(specifier, head, label="XP")
  else:
    return InternalBinaryNode(specifier,
                              InternalBinaryNode(head, complement, label="X'"),
                              label="XP")
