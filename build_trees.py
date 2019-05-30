from typing import List, Text, Tuple, Union


class InternalBinaryNode:

  def __init__(self, left_child, right_child, label=None):
    self.left_child = left_child
    self.right_child = right_child
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
                               self._child_to_latex(self.left_child, True),
                               self._child_to_latex(self.right_child, True))
    else:
      return "[ %s %s ]" % (self._child_to_latex(self.left_child),
                            self._child_to_latex(self.right_child))

  @staticmethod
  def _child_to_evalb(child):
    if isinstance(child, InternalBinaryNode):
      return child.to_evalb()
    else:
      return "(P %s)" % str(child)

  def to_evalb(self):
    """Export the tree to PARSE EVAL format."""
    return "(X %s %s)" % (self._child_to_evalb(self.left_child),
                          self._child_to_evalb(self.right_child))

  @classmethod
  def to_nested_lists(cls, tree):
    """Exported this tree to unlabelled nested lists of children."""
    if isinstance(tree, cls):
      return [cls.to_nested_lists(tree.left_child),
              cls.to_nested_lists(tree.right_child)]
    else:
      return tree


BinaryTree = Union[InternalBinaryNode, Text]


def greedy_parse(scored_tokens: List[Tuple[Text, float]]) -> BinaryTree:
  """This function greedily splits a tree by splitting at the greatest distances.

  To be more precise, this sentence splits recursively at the greatest distance
  and builds a constituent:

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
