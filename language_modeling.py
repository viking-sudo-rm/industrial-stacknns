import logging
import random

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
import torch

from data_readers.brown import BrownDatasetReader
from data_readers.linzen import LinzenLMDatasetReader
from data_readers.wsj import WSJDatasetReader
from stack_rnn_LM import StackRNNLanguageModel
from simple_rnn_LM import SimpleRNNLanguageModel


def main():
    # reader = BrownDatasetReader()
    # train_dataset = reader.read("data/brown.txt")
    # vocab = Vocabulary.from_instances(train_dataset)
    # dataset_name = "brown"

    reader = WSJDatasetReader(max_num=5000)
    train_dataset = reader.read("data/treebank_3/raw/wsj")
    vocab = Vocabulary.from_instances(train_dataset)
    dataset_name = "wsj"

    # reader = LinzenLMDatasetReader()
    # train_dataset = reader.read("StackNN/data/linzen/rnn_agr_simple/numpred.train")
    # validation_dataset = reader.read("StackNN/data/linzen/rnn_agr_simple/numpred.val")
    # vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    # dataset_name = "linzenlm"

    swap_push_pop = True

    # if torch.cuda.is_available():
    #     logging.info("Found and using CUDA.")
    #     device = 0
    # else:
    #     logging.info("No CUDA found.")
    #     device = -1

    # For some reason, this code runs MUCH slower on the GPU.
    device = -1

    model = StackRNNLanguageModel(vocab,
                                  rnn_dim=100,
                                  stack_dim=16,
                                  swap_push_pop=swap_push_pop,
                                  device=None if device == -1 else device)

    optimizer = torch.optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=16,
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    if device == 0:
        # Best practice to call this after constructing optimizer.
        model.cuda(device)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=5,
                      # validation_dataset=validation_dataset,
                      # patience=5
                      cuda_device=device)
    trainer.train()

    filename = dataset_name
    if isinstance(model, StackRNNLanguageModel):
        filename = "stack-" + filename
    if swap_push_pop:
        filename += "-swap"

    with open("saved_models/%s.th" % filename, "wb") as fh:
        torch.save(model.state_dict(), fh)
    vocab.save_to_files("saved_models/vocabulary-%s" % dataset_name)


if __name__ == "__main__":
    main()
