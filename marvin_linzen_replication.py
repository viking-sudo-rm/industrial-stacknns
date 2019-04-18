import logging
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
import torch

from data_readers.marvin_linzen import MarvinLinzenLMDatasetReader
from simple_rnn import SimpleRNNAgreementPredictor
from stack_rnn import StackRNNAgreementPredictor
from stack_rnn_LM import StackRNNLanguageModel


def main():
  reader = MarvinLinzenLMDatasetReader(append_null=False)
  train_dataset = reader.read(
      "StackNN/data/linzen/rnn_agr_simple/numpred.train")
  validation_dataset = reader.read(
      "StackNN/data/linzen/rnn_agr_simple/numpred.val")
  vocab = Vocabulary.from_files("saved_models/vocabulary_brown")

  model = StackRNNLanguageModel(vocab, rnn_dim=100, rnn_cell_type=torch.nn.GRUCell)
  model.load_state_dict(torch.load("saved_models/stack-brown.th"))

  optimizer = torch.optim.Adam(model.parameters())
  iterator = BucketIterator(batch_size=16, sorting_keys=[
      ("sentence", "num_tokens")])
  iterator.index_with(vocab)

  trainer = Trainer(model=model,
                    optimizer=optimizer,
                    iterator=iterator,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    patience=5)
  trainer.train()

  with open("/tmp/model.th", "wb") as fh:
    torch.save(model.state_dict(), fh)
  vocab.save_to_files("/tmp/vocabulary")


if __name__ == "__main__":
  main()
