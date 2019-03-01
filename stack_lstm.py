from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import Average, BooleanAccuracy
import torch

from StackNN.structs import Stack


class StackLSTMAgreementPredictor(Model):

    def __init__(self,
                 vocab,
                 num_embeddings=10000,
                 embedding_dim=50,
                 lstm_dim=650):

        super().__init__(vocab)
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

        self._lstm_dim = lstm_dim
        self._lstm_cell = torch.nn.LSTMCell(embedding_dim + lstm_dim, lstm_dim)
        self._stack_module = torch.nn.Linear(lstm_dim, 2)
        self._classifier = torch.nn.Linear(lstm_dim, 1)

        self._accuracy = BooleanAccuracy()
        self._pop_strength = Average()
        self._criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, sentence, label):
        embedded = self._embedder(sentence)
        batch_size = embedded.size(0)
        sentence_length = embedded.size(1)

        h, c = torch.zeros([batch_size, self._lstm_dim]), torch.zeros([batch_size, self._lstm_dim])
        stack = Stack(batch_size, self._lstm_dim)
        stack_summary = torch.zeros([batch_size, self._lstm_dim])

        for t in range(sentence_length):
            features = torch.cat([embedded[:, t], stack_summary], 1)
            h, c = self._lstm_cell(features, [h, c])
            stack_params = torch.sigmoid(self._stack_module(h))
            push_strengths = stack_params[:, 0]
            pop_strengths = stack_params[:, 1]
            self._pop_strength(torch.mean(pop_strengths))
            stack_summary = stack(h, push_strengths, pop_strengths)

        logits = torch.squeeze(self._classifier(h))
        prediction = (logits > 0.).float()
        label = label.float()
        loss = self._criterion(logits, label)
        accuracy = self._accuracy(prediction, label)

        return {
            "prediction": prediction,
            "accuracy": accuracy,
            "loss": loss,
        }
        # TODO: Custom metric: pop strength.

    def get_metrics(self, reset):
        return {
            "accuracy": self._accuracy.get_metric(reset),
            "pop_strength": self._pop_strength.get_metric(reset),
        }
