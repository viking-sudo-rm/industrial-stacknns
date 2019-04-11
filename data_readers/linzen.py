from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.tokenizers import Token


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


class LinzenLMDatasetReader(DatasetReader):

  """Dataset reader for Linzen as a language modeling task."""

  def __init__(self):
    super().__init__(lazy=False)
    self.token_indexers = {"tokens": SingleIdTokenIndexer()}

  def _read(self, file_path):
    with open(file_path) as f:
      for line in f:
        components = line.split("\t")
        sentence = components[0].split(' ')

        sentence_field = TextField([Token(word) for word in sentence[:-1]], self.token_indexers)

        label = SequenceLabelField(sentence[1:], sequence_field=sentence_field)

        yield Instance({
          "sentence": sentence_field,
          "label": label
        })