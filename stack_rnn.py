from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Average, BooleanAccuracy
import torch
import torch.nn.functional as F

from StackNN.structs import Stack
from StackNN.control_layer import ControlLayer


class StackRNNAgreementPredictor(Model):

    def __init__(self,
                 vocab,
                 num_embeddings=10000,
                 embedding_dim=50,
                 rnn_dim=650,
                 stack_dim=16,
                 rnn_cell_type=torch.nn.LSTMCell,
                 push_rnn_state=False):

        super().__init__(vocab)
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

        self._rnn_dim = rnn_dim
        self._stack_dim = stack_dim
        self._push_rnn_state = push_rnn_state

        self._rnn_cell = rnn_cell_type(embedding_dim + stack_dim, rnn_dim)
        self._control_layer = ControlLayer(rnn_dim, stack_dim, vision=4)
        self._classifier = torch.nn.Linear(rnn_dim, 1)

        self._accuracy = BooleanAccuracy()
        self._pop_strength = Average()
        self._criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, sentence, label=None):
        embedded = self._embedder(sentence)
        batch_size = embedded.size(0)
        sentence_length = embedded.size(1)

        h, c = torch.zeros([batch_size, self._rnn_dim]), torch.zeros([batch_size, self._rnn_dim])
        stack = Stack(batch_size, self._stack_dim)
        stack_summary = torch.zeros([batch_size, self._stack_dim])

        for t in range(sentence_length):
            features = torch.cat([embedded[:, t], stack_summary], 1)

            if isinstance(self._rnn_cell, torch.nn.LSTMCell):            
                h, c = self._rnn_cell(features, [h, c])
            else:
                h = self._rnn_cell(features, h)

            # Can push either stack vectors or hidden state onto the stack.
            instructions = self._control_layer(h)
            self._pop_strength(torch.mean(instructions.push_strengths - instructions.pop_strengths))
            if self._push_rnn_state:
                instructions.push_vectors = h
            stack_summary = stack(*instructions.make_tuple())

        logits = torch.squeeze(self._classifier(h))
        prediction = (logits > 0.).float()

        results = {
            "prediction": prediction,
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

    def get_metrics(self, reset):
        return {
            "accuracy": self._accuracy.get_metric(reset),
            "push_pop_strength": self._pop_strength.get_metric(reset),
        }
