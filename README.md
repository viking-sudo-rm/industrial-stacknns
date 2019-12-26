# Industrial Stack Neural Networks

This repository contains the code from the paper *Finding Syntactic Representations in Neural Stacks*, presented at BlackBox NLP 2019. Building on prior work, this paper analyzes the ability of stack RNNs to represent the compositional structure of natural language. We implement several stack RNN models for language modeling and agreement prediction, as well as a method for extracting unsupervised parse trees from trained stack RNN models. If you are inspired by this project, please cite:

```
@inproceedings{merrill-etal-2019-finding,
    title = "Finding Hierarchical Structure in Neural Stacks Using Unsupervised Parsing",
    author = "Merrill, William  and
      Khazan, Lenny  and
      Amsel, Noah  and
      Hao, Yiding  and
      Mendelsohn, Simon  and
      Frank, Robert",
    booktitle = "Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-4823",
    pages = "224--232",
    abstract = "Neural network architectures have been augmented with differentiable stacks in order to introduce a bias toward learning hierarchy-sensitive regularities. It has, however, proven difficult to assess the degree to which such a bias is effective, as the operation of the differentiable stack is not always interpretable. In this paper, we attempt to detect the presence of latent representations of hierarchical structure through an exploration of the unsupervised learning of constituency structure. Using a technique due to Shen et al. (2018a,b), we extract syntactic trees from the pushing behavior of stack RNNs trained on language modeling and classification objectives. We find that our models produce parses that reflect natural language syntactic constituencies, demonstrating that stack RNNs do indeed infer linguistically relevant hierarchical structure.",
}
```
