# Regularized inverse Scattering

This repository contains the code to reproduce the experiments of the paper:

[Generative networks as inverse problems with Scattering transforms](https://openreview.net/pdf?id=r1NYjfbR-)

Specifically, it contains the code necessary to invert the representations given by a fixed encoder (or embedding operator).

It is implemented using [PyTorch](http://pytorch.org/). The file GSN.py contains the class GSN that implements the optimization of a network defined in generator_architecture.py; in the file main.py one can modify all the parameters taking into account the following:

- There should be a 'datasets' folder in your home folder which contains two folders for the train and test images and two folders for its corresponding representations (embedding_attribute in main.py).

- The models and the generations are saved in the folder 'experiments/gsn_hf' inside your home folder, the name of the folder is the name of the experiment indicated as a parameter in main.py.

To compute the representations you can use [PyScatWave](https://github.com/edouardoyallon/pyscatwave) and to whiten them you can use PCA from [scikit-learn](http://scikit-learn.org).
