from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
import torch
import torch.nn.functional as F

from StackNN.structs import Stack
from StackNN.control_layer import ControlLayer


class StackRNNLanguageModel(Model):

    def __init__(self,
                 vocab,
                 embedding_dim=50,
                 stack_dim=16,
                 push_ones=True,
                 rnn_dim=650,
                 num_embeddings=None,  # Backward compatibility.
                 swap_push_pop=True,
                 rnn_cell_type=torch.nn.LSTMCell):

        super().__init__(vocab)
        self._vocab_size = vocab.get_vocab_size()
        if num_embeddings is None: num_embeddings = self._vocab_size
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

        self._stack_dim = stack_dim
        self._rnn_dim = rnn_dim

        self._push_ones = push_ones
        self._swap_push_pop = swap_push_pop

        if rnn_cell_type is not None:
            self._rnn_cell = rnn_cell_type(embedding_dim + stack_dim, rnn_dim)
        else:
            self._rnn_cell = None
            self.feedforward = torch.nn.Sequential(
                                            torch.nn.Linear(embedding_dim + rnn_dim, rnn_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(rnn_dim, rnn_dim),
                                            torch.nn.ReLU())

        self._control_layer = ControlLayer(rnn_dim, stack_dim, vision=4)
        self._classifier = torch.nn.Linear(rnn_dim, self._vocab_size)

        self._accuracy = CategoricalAccuracy()
        self._criterion = torch.nn.CrossEntropyLoss()
        self.instruction_history = None

    def forward(self, sentence, label=None):
        mask = get_text_field_mask(sentence)
        embedded = self._embedder(sentence)
        batch_size = embedded.size(0)
        sentence_length = embedded.size(1)

        h, c = torch.zeros([batch_size, self._rnn_dim]), torch.zeros([batch_size, self._rnn_dim])
        stack = Stack(batch_size, self._stack_dim)
        stack_summary = torch.zeros([batch_size, self._stack_dim])

        h_all_words = []
        instructions_list = []
        stack_total_strengths = []

        for t in range(sentence_length): # can't predict the next tag when you're at the last tag
            features = torch.cat([embedded[:, t], stack_summary], 1)

            if isinstance(self._rnn_cell, torch.nn.LSTMCell):
                h, c = self._rnn_cell(features, [h, c])
            elif isinstance(self._rnn_cell, torch.nn.GRUCell):
                h = self._rnn_cell(features, h)
            elif self._rnn_cell is None:
                h = self.feedforward(features)
            else:
                raise NotImplementedError

            instructions = self._control_layer(h)
            if self._push_ones:
                instructions.push_strengths = torch.ones_like(instructions.push_strengths)
            if self._swap_push_pop:
                temp = instructions.push_strengths
                instructions.push_strengths = instructions.pop_strengths
                instructions.pop_strengths = temp
            stack_summary = stack(*instructions.make_tuple())

            stack_total_strengths.append(sum(stack._strengths))
            h_all_words.append(h)
            instructions_list.append(instructions)

        stacked_h = torch.stack(h_all_words, dim=1)
        logits = self._classifier(stacked_h)
        predictions = torch.argmax(logits, dim=2).float()

        push_strengths = torch.stack([instr.push_strengths for instr in instructions_list], dim=-1)
        pop_strengths = torch.stack([instr.pop_strengths for instr in instructions_list], dim=-1)
        read_strengths = torch.stack([instr.read_strengths for instr in instructions_list], dim=-1)
        pop_dists = torch.stack([instr.pop_distributions for instr in instructions_list], dim=-2)
        read_dists = torch.stack([instr.read_distributions for instr in instructions_list], dim=-2)
        stack_total_strengths = torch.stack(stack_total_strengths, dim=-1)

        results = {
            "predictions": predictions,
            "push_strengths": push_strengths,
            "read_strengths": read_strengths,
            "pop_strengths": pop_strengths,
            "pop_dists": pop_dists,
            "read_dists": read_dists,
            "stack_total_strengths": stack_total_strengths,
        }

        if label is not None:
            self._accuracy(logits, label, mask)
            loss = sequence_cross_entropy_with_logits(logits, label, mask)
            results["loss"] = loss

        return results

    def get_metrics(self, reset):
        return {
            "accuracy": self._accuracy.get_metric(reset),
        }
