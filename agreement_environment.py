import random

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token


class LinzenEnvironment:

  def __init__(self, sentence, label):
    self._sentence = list(sentence)
    self._output = None
    self._label = label
    self._char_i = -1
    self._stack = []
    self.actions = self._make_actions()
    self._correct_predictions = 0

  @property
  def output(self):
    return "{} ({})".format(str(self._output), str(self._label))

  @property
  def accuracy(self):
    # for now, there is only one label per sentence, so there is only
    # one prediction to be made
    return int(self._correct_predictions > 0)

  """Generic declarations for types of actions."""

  def _pass_action(self):
    return 0., False

  def _make_push_action(self, value):

    def _push_action():
      self._stack.insert(0, value)
      # reward = -0.05 * len(self._stack)
      reward = 0
      return reward, False

    return _push_action

  def _pop_action(self):
    if len(self._stack) > 0:
      self._stack.pop(0)
      return 0., False
    return 0, False

  def _swap_action(self):
    if len(self._stack) > 0:
      self._stack[0] = 1 - self._stack[0]
      return 0., False
    return 0, False

  def _output_action(self):
    if len(self._stack) > 0:
      self._output = self._stack[0]
      if self._output == self._label:
        self._correct_predictions += 1
        reward = 1
      else:
        reward = 0
      return reward, True
    else:
      return 0., True

  """Specific mapping of integers to actions."""

  def _make_actions(self):
    return [
        self._make_push_action(0),
        self._make_push_action(1),
        self._pop_action,
        self._pass_action,
        self._swap_action,
        self._output_action,
    ]

  """Getting new inputs."""

  def observe_environment(self):
    """Get an incomplete observation of the input and stack state.

    Returns:
        Next input token.
        The top value on the stack.
    """
    return self._observe_next_char(), self._observe_stack()

  def _observe_next_char(self):
    """Return next character or None if we have reached the end of the sentence."""
    self._char_i += 1
    if self._char_i < len(self._sentence):
      return self._sentence[self._char_i]

  def _observe_stack(self):
    """Return 0, 1, or None depending on the top of the stack."""
    if len(self._stack) > 0:
      return self._stack[0]


@DatasetReader.register("linzen_dataset_reader")
class LinzenDatasetReader(DatasetReader):
  def __init__(self, append_null=True):
    super().__init__(lazy=False)
    self.token_indexers = {"tokens": SingleIdTokenIndexer()}
    self._append_null = append_null

  def _read(self, file_path):
    with open(file_path) as f:
      for line in f:
        label = 1 if line[:3] == 'VBP' else 0
        raw_sent = line[4:].strip().split(" ")
        sent = [Token(word) for word in raw_sent]
        if self._append_null:
          sent.append(Token("#"))
        yield Instance({"sentence": TextField(sent, self.token_indexers), "label": LabelField(str(label))})


class LinzenDataset:
  def __init__(self, filename):
    self.reader = LinzenDatasetReader()
    self.dataset = self.reader.read(filename)
    self.vocab = Vocabulary.from_instances(self.dataset)
    self.dataset_list = list(iter(self.dataset))
    self.instance = None
    self._label = None

  def get_env(self):
    idx = random.randint(0, len(self.dataset_list) - 1)
    self.instance = self.dataset_list[idx]
    sentence = [self.vocab.get_token_index(
        str(token)) for token in self.instance["sentence"]]
    self._label = int(self.instance["label"].label)
    return LinzenEnvironment(sentence, self._label)

  @property
  def input_string(self):
    return ' '.join(token.text for token in self.instance["sentence"])

  @property
  def label(self):
    return str(self._label)
