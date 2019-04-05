from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Average, BooleanAccuracy
import torch
import torch.nn.functional as F

from StackNN.structs import Stack
from StackNN.control_layer import ControlLayer


class StackRNNLanguageModel(Model):

    def __init__(self,
                 vocab,
                 num_embeddings=10000,
                 embedding_dim=None,
                 stack_dim=16,
                 rnn_dim=650,
                 rnn_cell_type=torch.nn.LSTMCell):

        super().__init__(vocab)
        self._vocab_size = vocab.get_vocab_size()
        if embedding_dim is None:
            embedding_dim = self._vocab_size
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

        self._stack_dim = stack_dim
        self._rnn_dim = rnn_dim

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

        self._accuracy = BooleanAccuracy()
        self._pop_strength = Average()
        self._criterion = torch.nn.CrossEntropyLoss()
        self.instruction_history = None

    def forward(self, sentence, label=None):
        embedded = self._embedder(sentence)
        batch_size = embedded.size(0)
        sentence_length = embedded.size(1)

        h, c = torch.zeros([batch_size, self._rnn_dim]), torch.zeros([batch_size, self._rnn_dim])
        stack = Stack(batch_size, self._stack_dim)
        stack_summary = torch.zeros([batch_size, self._stack_dim])

        h_all_words = []
        instructions_list = []

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
            self._pop_strength(torch.mean(instructions.push_strengths - instructions.pop_strengths))
            stack_summary = stack(*instructions.make_tuple())

            h_all_words.append(h)
            instructions_list.append(instructions)

        stacked_h = torch.stack(h_all_words, dim=1)
        logits = self._classifier(stacked_h)
        predictions = torch.argmax(logits, dim=2).float()

        pop_strengths = torch.stack([instr.pop_strengths for instr in instructions_list], dim=-1)

        results = {
            "predictions": predictions,
            "pop_strengths": pop_strengths,
        }

        if label is not None:
            self._accuracy(predictions.reshape(-1), label.reshape(-1).float())
            loss = self._criterion(logits.reshape(-1, self._vocab_size), label.reshape(-1))
            results["loss"] = loss

        return results

    def get_metrics(self, reset):
        return {
            "accuracy": self._accuracy.get_metric(reset),
            "push_pop_strength": self._pop_strength.get_metric(reset),
        }
