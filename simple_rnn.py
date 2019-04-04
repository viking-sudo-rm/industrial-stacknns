import torch

from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import BooleanAccuracy

@Model.register("simple_agreement_model")
class SimpleRNNAgreementPredictor(Model):

    def __init__(self,
                 vocab,
                 num_embeddings=10000,
                 embedding_dim=50,
                 rnn_dim=650,
                 rnn_type=torch.nn.LSTM):

        super().__init__(vocab)
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

        self._rnn = rnn_type(embedding_dim, rnn_dim, batch_first=True)
        self._linear = torch.nn.Linear(rnn_dim, 1)

        self._accuracy = BooleanAccuracy()
        self._criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, sentence, label):
        embedded = self._embedder(sentence)

        if isinstance(self._rnn, torch.nn.LSTM):
            _, (final_rnn_state, _) = self._rnn(embedded)
        else:
            _, final_rnn_state = self._rnn(embedded)

        final_rnn_state = torch.transpose(final_rnn_state, 0, 1)
        logits = self._linear(final_rnn_state)

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

    def get_metrics(self, reset=False):
        return {"accuracy": self._accuracy.get_metric(reset)}
