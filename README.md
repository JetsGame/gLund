[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3384920.svg)](https://doi.org/10.5281/zenodo.3384920)

gLund
======

This repository contains the code and results presented in [arXiv:1909.01359](https://arxiv.org/abs/1909.01359).

## About

gLund is a framework using the Lund jet plane to construct generative
models for jet substructure.

## Install gLund

gLund is tested and supported on 64-bit systems running Linux.

Install gLund with Python's pip package manager:
```
git clone https://github.com/JetsGame/gLund.git
cd gLund
pip install .
```
To install the package in a specific location, use
the "--target=PREFIX_PATH" flag.

This process will copy the `glund` program to your environment python path.

We recommend the installation of the gLund package using a `miniconda3`
environment with the
[configuration specified here](https://github.com/JetsGame/gLund/blob/master/environment.yml).

gLund requires the following packages:
- python3
- numpy
- [fastjet](http://fastjet.fr/) (compiled with --enable-pyext)
- matplotlib
- pandas
- keras
- tensorflow
- json
- gzip
- argparse
- scikit-image
- scikit-learn
- hyperopt (optional)

## Pre-trained models

The final models presented in
[arXiv:1909.01359](https://arxiv.org/abs/1909.01359 "gLund paper")
are stored in:
- results/lsgan: gLund LSGAN model trained on QCD jets (Pythia 8 + Delphes v3.4.1 fast detector simulation).
- results/vae: gLund VAE model trained on QCD jets (Pythia 8 + Delphes v3.4.1 fast detector simulation).
- results/wgangp: gLund WGAN-GP model trained on QCD jets (Pythia 8 + Delphes v3.4.1 fast detector simulation).

## Input data

All data used for the final models can be downloaded from the git-lfs repository
at https://github.com/JetsGame/data.

## Running the code

In order to launch the code run:
```
glund --output <output_folder>  <runcard.yaml>
```
This will create a folder containing the result of the fit.

To create new samples from an existing model, as well as some diagnostic plots, use
```
glund_generate --save --ngen <number_to_generate> --output <result_file.npy> <model>
```

## References

* S. Carrazza and F. A. Dreyer, "Lund jet images from generative and cycle-consistent adversarial networks,"
  [arXiv:1909.01359](https://arxiv.org/abs/1909.01359 "gLund paper")
