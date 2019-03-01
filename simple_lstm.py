import torch

from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import BooleanAccuracy


class SimpleLSTMAgreementPredictor(Model):

    def __init__(self,
                 vocab,
                 num_embeddings=10000,
                 embedding_dim=50,
                 lstm_dim=650):

        super().__init__(vocab)
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

        self._lstm = torch.nn.LSTM(embedding_dim, lstm_dim, batch_first=True)
        self._linear = torch.nn.Linear(lstm_dim, 1)

        self._accuracy = BooleanAccuracy()
        self._criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, sentence, label):
        embedded = self._embedder(sentence)
        _, (final_lstm_state, _) = self._lstm(embedded)
        final_lstm_state = torch.transpose(final_lstm_state, 0, 1)
        logits = self._linear(final_lstm_state)

        logits = torch.squeeze(logits)
        prediction = (logits > 0.).float()
        label = label.float()
        accuracy = self._accuracy(prediction, label)
        loss = self._criterion(logits, label)

        return {
            "prediction": prediction,
            "accuracy": accuracy,
            "loss": loss,
        }

    def get_metrics(self, reset):
        return {"accuracy": self._accuracy.get_metric(reset)}
