[![DOI](https://zenodo.org/badge/DOI/)](https://doi.org)

gLund
======

This repository contains the code and results presented in [arXiv:1908.xxxxx](https://arxiv.org/abs/1908.xxxxxx).

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
[arXiv:1908.xxxxx](https://arxiv.org/abs/1908.xxxxx "gLund paper")
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

To create diagnostic plots from an existing model, use
```
glund_plot_samples --data output_folder/generated_images.npy  --reference ../data/valid/valid_QCD_500GeV.json.gz
```

## References

* S. Carrazza and F. A. Dreyer, "Towards a generative model for jet substructure,"
  [arXiv:1908.xxxxx](https://arxiv.org/abs/1908.xxxxx "gLund paper")
