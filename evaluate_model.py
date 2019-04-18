import sys
from simple_rnn import SimpleRNNAgreementPredictor
from stack_rnn import StackRNNAgreementPredictor
from agreement_environment import LinzenDatasetReader
from allennlp.data.vocabulary import Vocabulary
import torch
from allennlp.data.iterators import BucketIterator
from allennlp.training.util import evaluate
# from allennlp.models.archival import archive_model


def main():
  reader = LinzenDatasetReader(append_null=False)
  vocab = Vocabulary.from_files("saved_models/vocabulary")

  stack = StackRNNAgreementPredictor(
      vocab, rnn_dim=100, rnn_cell_type=torch.nn.GRUCell)
  stack.load_state_dict(torch.load("saved_models/stack-linzen.th"))

  lstm = SimpleRNNAgreementPredictor(vocab, rnn_dim=18, rnn_type=torch.nn.GRU)
  lstm.load_state_dict(torch.load("saved_models/lstm-linzen.th"))

  iterator = BucketIterator(batch_size=32, sorting_keys=[
      ("sentence", "num_tokens")])
  iterator.index_with(vocab)

  dataset = reader.read("StackNN/data/linzen/rnn_agr_simple/numpred.test")
  stack_metrics = evaluate(stack, dataset, iterator, -1, "")
  lstm_metrics = evaluate(stack, dataset, iterator, -1, "")
  print(stack_metrics)
  print(lstm_metrics)

  for i in range(6):
    dataset = reader.read(
      "StackNN/data/linzen/rnn_agr_simple/numpred.test." + str(i))
    stack_metrics = evaluate(stack, dataset, iterator, -1, "")
    lstm_metrics = evaluate(lstm, dataset, iterator, -1, "")
    print(stack_metrics)
    print(lstm_metrics)


if __name__ == "__main__":
  main()
