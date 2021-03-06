from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class BrownDatasetReader(DatasetReader):

    """A reader for the Brown POS-tagging corpus.

    To download the data, make sure you are in the industrial-stacknns/ directory, and run:

    wget -O data/brown.txt http://www.sls.hawaii.edu/bley-vroman/browntag_nolines.txt
    """

    def __init__(self, words=False, labels=True):
        super().__init__(lazy=False)
        self._words = words
        self._labels = labels
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        #self._tag_indexers = {"tags": SingleIdTokenIndexer()}

    def _read(self, file_path):
        with open(file_path) as f:
            for line in f:
                raw_sent = line.strip().split(" ")
                if len(raw_sent) < 2:
                    continue
                token_idx = 0 if self._words else 1
                tokens = [raw_pair.split("_")[token_idx] for raw_pair in raw_sent]
                yield self.text_to_instance(tokens)

    def text_to_instance(self, text):
        if self._labels:
            sentence = TextField([Token(word) for word in text[:-1]],
                                 self._token_indexers)
            labels = SequenceLabelField(text[1:], sequence_field=sentence)
            return Instance({
                "sentence": sentence,
                "label": labels,
            })
        else:
            sentence = TextField([Token(word) for word in text],
                                 self._token_indexers)
            return Instance({
                "sentence": sentence,
            })
