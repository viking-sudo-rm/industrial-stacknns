from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Average, BooleanAccuracy
import torch
import torch.nn.functional as F

from StackNN.structs import Stack
from StackNN.control_layer import ControlLayer

@Model.register("stack_agreement_model")
class StackRNNAgreementPredictor(Model):

  def __init__(self,
               vocab,
               num_embeddings=None, # Backwards compatibility.
               embedding_dim=50,
               rnn_dim=650,
               stack_dim=16,
               rnn_cell_type=torch.nn.LSTMCell,
               push_rnn_state=False,
               swap_push_pop=True, # Backward compatibility.
               push_ones=True):

    super().__init__(vocab)
    self._vocab_size = vocab.get_vocab_size()
    if num_embeddings is None: num_embeddings = self._vocab_size
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

    self._rnn_dim = rnn_dim
    self._stack_dim = stack_dim
    self._push_rnn_state = push_rnn_state

    if rnn_cell_type == "gru":
      rnn_cell_type = torch.nn.GRUCell

    self._rnn_cell = rnn_cell_type(embedding_dim + stack_dim, rnn_dim)
    self._control_layer = ControlLayer(rnn_dim, stack_dim, vision=4)
    self._classifier = torch.nn.Linear(rnn_dim, 1)

    self._accuracy = BooleanAccuracy()
    self._pop_strength = Average()
    self._criterion = torch.nn.BCEWithLogitsLoss()

    self._push_ones = push_ones
    self._swap_push_pop = swap_push_pop

  def forward(self, sentence, label=None):
    embedded = self._embedder(sentence)
    batch_size = embedded.size(0)
    sentence_length = embedded.size(1)

    h, c = torch.zeros([batch_size, self._rnn_dim]), torch.zeros(
        [batch_size, self._rnn_dim])
    stack = Stack(batch_size, self._stack_dim)
    stack_summary = torch.zeros([batch_size, self._stack_dim])

    instructions_list = []
    stack_total_strengths = []

    for t in range(sentence_length):
      features = torch.cat([embedded[:, t], stack_summary], 1)

      if isinstance(self._rnn_cell, torch.nn.LSTMCell):
        h, c = self._rnn_cell(features, [h, c])
      else:
        h = self._rnn_cell(features, h)

      # Can push either stack vectors or hidden state onto the stack.
      instructions = self._control_layer(h)
      if self._push_ones:
        instructions.push_strengths = torch.ones_like(instructions.push_strengths)
      if self._swap_push_pop:
        temp = instructions.push_strengths
        instructions.push_strengths = instructions.pop_strengths
        instructions.pop_strengths = temp
      # self._pop_strength(torch.mean(
          # instructions.push_strengths - instructions.pop_strengths))
      if self._push_rnn_state:
        instructions.push_vectors = h
      stack_summary = stack(*instructions.make_tuple())

      stack_total_strengths.append(sum(stack._strengths))
      instructions_list.append(instructions)

    logits = torch.squeeze(self._classifier(h))
    prediction = (logits > 0.).float()

    push_strengths = torch.stack([instr.push_strengths for instr in instructions_list], dim=-1)
    pop_strengths = torch.stack([instr.pop_strengths for instr in instructions_list], dim=-1)
    read_strengths = torch.stack([instr.read_strengths for instr in instructions_list], dim=-1)
    pop_dists = torch.stack([instr.pop_distributions for instr in instructions_list], dim=-2)
    read_dists = torch.stack([instr.read_distributions for instr in instructions_list], dim=-2)
    stack_total_strengths = torch.stack(stack_total_strengths, dim=-1)

    results = {
        "prediction": prediction,
        "instructions": instructions_list,
        "push_strengths": push_strengths,
        "read_strengths": read_strengths,
        "pop_strengths": pop_strengths,
        "pop_dists": pop_dists,
        "read_dists": read_dists,
        "stack_total_strengths": stack_total_strengths,
    }

    if label is not None:
      label = label.float()
      loss = self._criterion(logits, label)
      accuracy = self._accuracy(prediction, label)
      results.update({
          "accuracy": accuracy,
          "loss": loss,
      })

    return results

  def get_metrics(self, reset=False):
    return {
        "accuracy": self._accuracy.get_metric(reset)
    }
