from overrides import overrides
import os

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token


class WSJDatasetReader(DatasetReader):

    """Loads from raw WSJ files."""

    def __init__(self, include_pos=False, max_num=2300):
        super().__init__(lazy=False)
        self._include_pos = include_pos
        self._max_num = max_num
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def _read(self, dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if not file.startswith("wsj_") or \
                        int(file[4:]) >= self._max_num:
                    continue
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line == ".START":
                            continue
                        yield self.text_to_instance(line.split(" "))

    @overrides
    def text_to_instance(self, text):
        sentence_field = TextField([Token(word)
                                    for word in text[:-1]],
                                   self.token_indexers)
        label = SequenceLabelField(text[1:],
                                   sequence_field=sentence_field)

        return Instance({
            "sentence": sentence_field,
            "label": label,
        })
