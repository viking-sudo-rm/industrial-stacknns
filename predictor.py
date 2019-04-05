from overrides import overrides

import torch
from allennlp.predictors.predictor import Predictor


class TreePredictor(Predictor):

    """Predict trees using a trained language model."""

    def __init__(self, model, dataset_reader):
        super().__init__(model, dataset_reader)
        self._model = model
        self._dataset_reader = dataset_reader

    def predict(self, sentence):
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict):
        print("_json_to_instance", json_dict)
        sentence = json_dict["sentence"]
        # TODO: Might want to use a real tokenizer here.
        tokens = sentence.split(" ")
        return self._dataset_reader.text_to_instance(tokens)
