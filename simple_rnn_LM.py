from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
import torch
import torch.nn.functional as F
import numpy as np


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
        self._perplexity = Average()
        self._criterion = torch.nn.CrossEntropyLoss()
        self.instruction_history = None

    def forward(self, sentence, label=None):
        mask = get_text_field_mask(sentence)
        embeddings = self._embedder(sentence)
        hidden_states, _ = self._rnn(embeddings)
        logits = self._classifier(hidden_states)
        
        # TODO: there must be a better way to do this?
        final_logits_mask = np.zeros((mask.shape[0],))
        for i in range(final_logits_mask.shape[0]):
            # get index of the last 1 in the mask for this row
            for j in reversed(range(mask.shape[1])):
                if mask[i][j] == 1:
                    final_logits_mask[i] = j
                    break

        final_logits = logits[np.arange(logits.shape[0]), final_logits_mask, :]
        # final_logits = logits[:,-1,:].squeeze()
        
        predictions = torch.argmax(logits, dim=2).float()

        results = {
            "predictions": predictions,
            "final_logits": final_logits
        }

        if label is not None:
            self._accuracy(logits, label, mask)
            loss = sequence_cross_entropy_with_logits(logits, label, mask)
            self._perplexity(np.exp(loss.detach().numpy()))
            results["loss"] = loss

        return results

    def get_metrics(self, reset=False):
        return {
            "accuracy": self._accuracy.get_metric(reset),
            "perplexity": self._perplexity.get_metric(reset)
        }
