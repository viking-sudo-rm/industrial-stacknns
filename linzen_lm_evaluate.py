import logging
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
import torch

from data_readers.brown import BrownDatasetReader
from data_readers.linzen import LinzenLMDatasetReader
from stack_rnn_LM import StackRNNLanguageModel
from utils import utils
import numpy as np
from allennlp.nn.util import get_text_field_mask


def main():
    reader = LinzenLMDatasetReader(test_mode=True)
    test_dataset = reader.read("data/agr_50_mostcommon_10K.tsv.10")
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")

    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16)
    model.load_state_dict(torch.load("saved_models/stack-linzen-swap.th"))

    iterator = BucketIterator(batch_size=16, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    inflect, _ = utils.gen_inflect_from_vocab()

    correct = total = 0
    for batch in iterator(test_dataset, num_epochs=1, shuffle=False):
        res = model.forward(batch["sentence"])
        logits = res["final_logits"]
        grammatical_labels = batch["label"]
        grammatical_words = [vocab.get_token_from_index(x.item()) for x in grammatical_labels]
        ungrammatical_words = [inflect[x] for x in grammatical_words]
        ungrammatical_labels = [vocab.get_token_index(x) for x in ungrammatical_words]
    
        for i in range(16):
            # print([vocab.get_token_from_index(x.item()) for x in batch["sentence"]["tokens"][i]])
            # print(grammatical_words[i], ungrammatical_words[i])
            # print(logits[i][grammatical_labels[i]].item(), logits[i][ungrammatical_labels[i]].item())
            if logits[i][grammatical_labels[i]] > logits[i][ungrammatical_labels[i]]:
                correct += 1

        total += 16
        print(correct/total)
    
    print(correct/total)

if __name__ == "__main__":
    main()
