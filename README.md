gLund
======

This repository contains the code and results presented in (....) .

## About

gLund is a framework using the Lund jet plane to construct generative 
models for jet substructure.

## Install gLund

gLund is tested and supported on 64-bit systems running Linux.

Install gLund with Python's pip package manager:
```
git clone https://github.com/JetsGames/gLund.git
cd gLund
pip install .
```
To install the package in a specific location, use
the "--target=PREFIX_PATH" flag.

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

* TBA
