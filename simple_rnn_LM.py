from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
import torch
import torch.nn.functional as F


class SimpleRNNLanguageModel(Model):

    def __init__(self,
                 vocab,
                 embedding_dim=50,
                 rnn_dim=650,
                 num_embeddings=None,  # Backward compatibility.
                 swap_push_pop=True,  # Backward compatibility.
                 rnn_type=torch.nn.LSTM):

        super().__init__(vocab)
        self._vocab_size = vocab.get_vocab_size()
        if num_embeddings is None: num_embeddings = self._vocab_size
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self._embedder = BasicTextFieldEmbedder({"tokens": embedding})

        self._rnn_dim = rnn_dim

        self._rnn = rnn_type(embedding_dim, rnn_dim, batch_first=True)
        self._classifier = torch.nn.Linear(rnn_dim, self._vocab_size)

        self._accuracy = CategoricalAccuracy()
        self._criterion = torch.nn.CrossEntropyLoss()
        self.instruction_history = None

    def forward(self, sentence, label=None):
        mask = get_text_field_mask(sentence)
        embeddings = self._embedder(sentence)
        hidden_states, _ = self._rnn(embeddings)
        logits = self._classifier(hidden_states)
        predictions = torch.argmax(logits, dim=2).float()

        results = {
            "predictions": predictions,
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
