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

    def __init__(self):
        super().__init__(lazy=False)
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._tag_indexers = {"tags": SingleIdTokenIndexer()}

    def _read(self, file_path):
        with open(file_path) as f:
            for line in f:
                if line == "\n":
                    continue
                raw_sent = line.strip().split(" ")
                pairs = [raw_pair.split("_") for raw_pair in raw_sent]
                sentence = TextField([Token(tag) for _, tag in pairs], self._tag_indexers)
                tags = SequenceLabelField([tag for _, tag in pairs], sequence_field=sentence)
                yield Instance({
                    "sentence": sentence,
                    "label": tags,
                })
