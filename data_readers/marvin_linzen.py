from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.tokenizers import Token

class MarvinLinzenLMDatasetReader(DatasetReader):

  """Dataset reader for Marvin-Linzen as a language modeling task."""

  def __init__(self):
    super().__init__(lazy=False)
    self.token_indexers = {"tokens": SingleIdTokenIndexer()}

  def _read(self, file_path):
    with open(file_path) as f:
      for line in f:
        components = line.split(',')
        if components[0] == "test case":
            continue

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
