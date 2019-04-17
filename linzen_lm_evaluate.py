import logging
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
import torch

from data_readers.brown import BrownDatasetReader
from data_readers.linzen import LinzenLMDatasetReader
from stack_rnn_LM import StackRNNLanguageModel
from simple_rnn_LM import SimpleRNNLanguageModel
from utils import utils
import numpy as np
from allennlp.nn.util import get_text_field_mask


PERPLEXITY = True

def main():
    reader = LinzenLMDatasetReader(test_mode=not PERPLEXITY)
    test_dataset = reader.read("data/agr_50_mostcommon_10K.tsv.10")
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")

    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16)
    model.load_state_dict(torch.load("saved_models/stack-linzen-swap.th"))

    iterator = BucketIterator(batch_size=16, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    inflect, _ = utils.gen_inflect_from_vocab()
    # print(inflect)

    correct = total = 0
    for batch in iterator(test_dataset, num_epochs=1, shuffle=False):
        res = model.forward(batch["sentence"], batch["label"])
        
        if not PERPLEXITY:
            logits = res["final_logits"]
            grammatical_labels = list(batch["grammatical_verb"])
            grammatical_words = [vocab.get_token_from_index(x.item(), namespace="labels") for x in grammatical_labels]
            ungrammatical_words = [inflect[x] for x in grammatical_words]
            for i in reversed(range(len(grammatical_words))):
                if ungrammatical_words[i] not in vocab.get_token_to_index_vocabulary(namespace="labels"):
                    del grammatical_labels[i]
                    del grammatical_words[i]
                    del ungrammatical_words[i]
            ungrammatical_labels = [vocab.get_token_index(x, namespace="labels") for x in ungrammatical_words]
            
            vbz_idx = vocab.get_token_index("VBZ")
            vbp_idx = vocab.get_token_index("VBP")
        
            for i in range(len(ungrammatical_labels)):
                # print([vocab.get_token_from_index(x.item()) for x in batch["sentence"]["tokens"][i]])
                # print(grammatical_words[i], ungrammatical_words[i], vocab.get_token_from_index(torch.argmax(logits[i]).item(), namespace="labels"))
                # print(logits[i, grammatical_labels[i]].item(), logits[i, ungrammatical_labels[i]].item(), torch.max(logits[i]).item())
                # print(logits[i, vbz_idx].item(), logits[i, vbp_idx].item())
                # print(logits.shape)
                if logits[i, grammatical_labels[i]] > logits[i, ungrammatical_labels[i]]:
                    correct += 1

            total += len(ungrammatical_labels)
            # print(correct/total)
    
    if PERPLEXITY:
        print(model.get_metrics())
        # lstm-linzen-15        
        # 'perplexity': 87.98544236728482, 'accuracy': 0.9393090689517839

        # lstm-linzen
        # 'perplexity': 91.69342339828427, 'accuracy': 0.939482508318666

        # stack-linzen
        # 'perplexity': 92.81089779237806, 'accuracy': 0.9359109420969461
    else:
        print(correct/total)

if __name__ == "__main__":
    main()
