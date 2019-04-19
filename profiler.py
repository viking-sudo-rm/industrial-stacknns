import matplotlib.pyplot as plt
import seaborn as sns

import torch
from pandas import DataFrame, concat

from allennlp.data.vocabulary import Vocabulary

from build_trees import greedy_parse
from data_readers.brown import BrownDatasetReader
from predictor import TreePredictor
from stack_rnn_LM import StackRNNLanguageModel
from stack_rnn import StackRNNAgreementPredictor

from build_trees import greedy_parse

def mybar(ser1, lab1, ser2, lab2, cat_label, ax=None):
    y_label = "Probability"
    n = len(ser1)
    df1 = DataFrame({"type": lab1,
                          y_label: ser1,
                          cat_label: list(range(len(ser1)))})
    df2 = DataFrame({"type": lab2,
                          y_label: ser2,
                          cat_label: list(range(len(ser2)))})
    sns.barplot(x=cat_label, y=y_label, hue="type", data=concat([df2, df1], ignore_index=True), ax=ax)

    '''   plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off'''

def firststyle(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,)      # ticks along the bottom edge are off
    ax.set_ylim((0.0, 1.0))
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    #ax.legend(loc=3)

def restyle(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,)      # ticks along the bottom edge are off
    ax.set_ylim((0.0, 1.0))
    ax.get_legend().remove()

def one_hist_wonder(results, sentence, fig):
    labels = ["{0}\n({1:.2f})".format(sentence[i],results["stack_total_strengths"][i]) for i in range(len(sentence))]
    push = DataFrame({"type": "Push Strength",
                          "Strength": results['push_strengths'],
                          "Word (Stack Total)": labels[0:]})
    pop = DataFrame({"type": "Pop Strength",
                          "Strength": results['pop_strengths'],
                          "Word (Stack Total)": labels[0:]})
    read = DataFrame({"type": "Read Strength",
                          "Strength": results['read_strengths'],
                          "Word (Stack Total)": labels[0:]})
    sns.barplot(x="Word (Stack Total)", y="Strength", hue="type", data=concat([push, read], ignore_index=True)) #ax = ax

def profile_sentence(results, sentence, fig, swap, offset=0):
    sentence_ix = 0
    plots = []

    margin=0.4
    width_per_plot = (1.0-margin)/(len(sentence)-1)

    #print("The label under each plot is the word being predicted at that step")
    print("Blue is {0} strength, orange is read strength".format("pop" if swap else "push"))
    fig.suptitle('Push (blue) and read (orange) distributions at each word\n', fontsize=12)
    for word_n in range(offset, len(sentence)):
        # instructs = results['instructions'][word_n]
        """
        # for many sentences
        push_strength = instructs.push_strengths.data[sentence_ix]
        pop_strength = instructs.pop_strengths.data[sentence_ix]
        read_strengths = instructs.read_strengths.data[sentence_ix]
        pop_distribution = instructs.pop_distributions.data[sentence_ix,:]
        read_distribution = instructs.read_distributions.data[sentence_ix,:]
        """
        """
        #for just one
        push_strength = instructs.push_strengths.data
        pop_strength = instructs.pop_strengths.data
        read_strengths = instructs.read_strengths.data
        pop_distribution = instructs.pop_distributions.data[sentence_ix,:]
        read_distribution = instructs.read_distributions.data[sentence_ix,:]
        """

        push_strength = results['push_strengths'][word_n]
        pop_strength = results['pop_strengths'][word_n]
        read_strengths = results['read_strengths'][word_n]
        pop_distribution = results['pop_dists'][word_n]
        read_distribution = results['read_dists'][word_n]
        stack_total_strength = results["stack_total_strengths"][word_n]

        ax1 = fig.add_axes([margin/2.0 + word_n*width_per_plot, margin, 0.8*width_per_plot, 5*width_per_plot],
                           ylim=(0, 1.0), xlabel=sentence[word_n])
        #label = "Read: {0:.2f}\nPop:{2:.2f}\nPush:{3:.2f}\n{4} ({1:.2f})".format(read_strengths,stack_total_strength, pop_strength, push_strength, sentence[word_n])
        label = "Read: {0:.2f}\nTotal:{1:.2f}\nPush:{2:.2f}\n{3}".format(read_strengths,stack_total_strength, push_strength, sentence[word_n])
        mybar(read_distribution, "Read Strength", pop_distribution, "Pop Strength", label, ax=ax1)
        restyle(ax1)
        plots.append(ax1)

    firststyle(plots[0])

def main():
    # language model, fixed pop
    swap = True
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16, num_embeddings=10030, swap_push_pop=swap)
    with open("saved_models/stack-linzen-swap.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    # classification model, fixed pop
    """swap = True
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    print(vocab.get_vocab_size())
    model = StackRNNAgreementPredictor(vocab, rnn_dim=100, stack_dim=16, rnn_cell_type=torch.nn.GRUCell, num_embeddings=10000, swap_push_pop=swap)
    with open("saved_models/stack-linzen-class.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))"""

    # language model fixed push
    """swap = False
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16, num_embeddings=10030, swap_push_pop=swap)
    with open("saved_models/stack-linzen.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))"""

    """swap=False
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNAgreementPredictor(vocab, rnn_dim=100, stack_dim=16, rnn_cell_type=torch.nn.GRUCell, num_embeddings=9968, push_ones=False, swap_push_pop=swap)
    with open("saved_models/stack-linzen-class-nopushpop.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))"""

    """swap=True
    vocab = Vocabulary.from_files("saved_models/vocabulary-linzen")
    model = StackRNNAgreementPredictor(vocab, rnn_dim=100, stack_dim=16, rnn_cell_type=torch.nn.GRUCell, num_embeddings=9968, swap_push_pop=swap)
    with open("saved_models/stack-linzen-class-pop.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))"""

    dataset_reader = BrownDatasetReader(labels=False) # true?
    predictor = TreePredictor(model, dataset_reader)

    sentence = "the man in the hospitals eats an apple"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    #one_hist_wonder(prediction, sentence.split(" "), fig)
    profile_sentence(prediction, sentence.split(" "), fig, swap)
    plt.show()

    sentence = "the cat that dogs chase eats apples"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    #one_hist_wonder(prediction, sentence.split(" "), fig)
    profile_sentence(prediction, sentence.split(" "), fig, swap)
    plt.show()

    sentence = "the man who likes eating apples is full"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    #one_hist_wonder(prediction, sentence.split(" "), fig)
    profile_sentence(prediction, sentence.split(" "), fig, swap)
    plt.show()

    """sentence = "dogs chase the cat"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    #one_hist_wonder(prediction, sentence.split(" "), fig)
    profile_sentence(prediction, sentence.split(" "), fig, swap)
    plt.show()

    sentence = "people ate apples"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    one_hist_wonder(prediction, sentence.split(" "), fig)
    #profile_sentence(prediction, sentence.split(" "), fig, swap)
    #plt.show()"""



if __name__ == "__main__":
    main()

    """swap = True
    vocab = Vocabulary.from_files("saved_models/vocabulary-brown")
    model = StackRNNLanguageModel(vocab, rnn_dim=100, stack_dim=16, num_embeddings=10000, swap_push_pop=swap)
    with open("saved_models/stack-brown-swap.th", "rb") as fh:
        model.load_state_dict(torch.load(fh))

    dataset_reader = BrownDatasetReader(labels=False) # true?
    predictor = TreePredictor(model, dataset_reader)

    sentence = "AT NN IN AT NNS VBD AT JJ NN"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    #one_hist_wonder(prediction, sentence.split(" "), fig)
    profile_sentence(prediction, sentence.split(" "), fig, swap)
    plt.show()

    sentence = "AT NNS VBD AT JJ JJ NN"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    #one_hist_wonder(prediction, sentence.split(" "), fig)
    profile_sentence(prediction, sentence.split(" "), fig, swap)
    plt.show()

    sentence = "AT NNS CC AT NN VBD IN AT NN"
    prediction = predictor.predict(sentence)
    fig = plt.figure()
    #one_hist_wonder(prediction, sentence.split(" "), fig)
    profile_sentence(prediction, sentence.split(" "), fig, swap)
    plt.show()"""
