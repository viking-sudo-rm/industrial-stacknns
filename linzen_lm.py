import logging
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
import torch

from data_readers.brown import BrownDatasetReader
from data_readers.linzen import LinzenLMDatasetReader
from stack_rnn_LM import StackRNNLanguageModel


def main():
    reader = LinzenLMDatasetReader()
    # download data from 
    # https://drive.google.com/file/d/1Yi2PThfLtlnFpN2oJu3dCYPDyv1CnGX8/view?usp=sharing
    # command used to generate train/test split (for reference)
    # `gawk 'BEGIN {srand()} {f = FILENAME (rand() <= 0.9 ? ".90" : ".10"); print > f}' agr_50_mostcommon_10K.tsv`
    train_dataset = reader.read("data/agr_50_mostcommon_10K.tsv.90")
    vocab = Vocabulary.from_instances(train_dataset)
    dataset_name = "linzen"
    swap = False

    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  swap_push_pop=swap)

    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16,
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=5)
    trainer.train()

    vocab.save_to_files("saved_models/vocabulary-%s" % dataset_name)
    dataset_name += "-swap" if swap else ""
    with open("saved_models/stack-%s.th" % dataset_name, "wb") as fh:
        torch.save(model.state_dict(), fh)


if __name__ == "__main__":
    main()
