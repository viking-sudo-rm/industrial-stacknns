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
    reader = BrownDatasetReader()
    train_dataset = reader.read("data/brown.txt")
    vocab = Vocabulary.from_instances(train_dataset)
    dataset_name = "brown"
    swap_push_pop = True

    # reader = LinzenLMDatasetReader()
    # train_dataset = reader.read("StackNN/data/linzen/rnn_agr_simple/numpred.train")
    # validation_dataset = reader.read("StackNN/data/linzen/rnn_agr_simple/numpred.val")
    # vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    # dataset_name = "linzenlm"

    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  swap_push_pop=swap_push_pop)

    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=5
                      # validation_dataset=validation_dataset,
                      # patience=5
                     )
    trainer.train()

    suffix = dataset_name + ("-swap" if swap_push_pop else "")
    with open("saved_models/stack-%s.th" % suffix, "wb") as fh:
        torch.save(model.state_dict(), fh)
    vocab.save_to_files("saved_models/vocabulary-%s" % suffix)


if __name__ == "__main__":
    main()
