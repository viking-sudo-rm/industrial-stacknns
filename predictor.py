import torch


class StackTaskPredictor:

    # TODO: Maybe this should extend allennlp.Predictor.

    def __init__(self, model, vocabulary):
        self._model = model
        self._vocabulary = vocabulary

    def predict(self, sentence):
        indices = [self._vocabulary.get_token_index(token) for token in sentence]
        tensor = torch.LongTensor([indices])
        return self._model({'tokens':tensor})
