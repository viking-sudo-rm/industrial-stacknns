#from agreement_environment import POSDatasetReader

import logging
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
import torch

from stack_rnn_LM import StackRNNLanguageModel


def main():
    reader = BrownDatasetReader(append_null=False)
    train_dataset = reader.read("data/rnn_agr_simple/numpred.train")
    validation_dataset = reader.read("data/rnn_agr_simple/numpred.val")
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    model = StackRNNLanguageModel(vocab, rnn_dim=8)

    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=20)
    trainer.train()

    with open("/tmp/model.th", "wb") as fh:
        torch.save(model.state_dict(), fh)
    vocab.save_to_files("/tmp/vocabulary")


if __name__ == "__main__":
    main()
