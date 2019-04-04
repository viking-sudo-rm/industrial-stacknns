import torch
from allennlp.data.vocabulary import Vocabulary

from stack_rnn_LM import StackRNNLanguageModel
from predictor import StackTaskPredictor

def brown_predict(sentence):
    vocab = Vocabulary.from_files("saved_models/vocabulary-brown")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16)

    with open("saved_models/stack-brown.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    predictor = StackTaskPredictor(model, vocab)

    #results = predictor.predict("AT NP NN JJ")
    results = predictor.predict(sentence)
    return results

def main():
    print(brown_predict(["AT", "NP", "NN", "JJ"]))

if __name__ == "__main__":
    main()
