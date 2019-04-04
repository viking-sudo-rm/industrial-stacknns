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


class LinzenDatasetLMReader(DatasetReader):

  """Dataset reader for Linzen as a language modeling task."""

  def __init__(self):
    super().__init__(lazy=False)
    self.token_indexers = {"tokens": SingleIdTokenIndexer()}

  def _read(self, file_path):
    with open(file_path) as f:
      for line in f:
        final_word = line[:3]
        word_prefix = line[4:].strip().split(" ")

        sentence = TextField([Token(word) for word in word_prefix], self.token_indexers)

        label_list = [word for word in word_prefix[1:]]
        label_list.append(final_word)
        label = SequenceLabelField(label_list, sequence_field=sentence)

        yield Instance({
            "sentence": sentence,
            "label": label,
          })
