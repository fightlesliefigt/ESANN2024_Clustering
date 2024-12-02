# Geometric Deep Learning to Enhance Imbalanced Domain Adaptation in EEG

This repository contains code accompanying the paper *Geometric Deep Learning to Enhance Imbalanced Domain Adaptation in EEG*.

## File list

The following files are provided in this repository:

`demo.ipynb` A jupyter notebook demonstrating Clustering can compensate the label shift in target domain in an unsupervised fashion.

`spdnets` A folder containing the whole geometric deep learning framework.

`pretrained_model` A folder containing the pretrained model for the source domains, enabling immediate source-free unsupervised domain adaptation.

## Requirements

All dependencies are managed with the `conda` package manager.
Please follow the user guide to [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) `conda`.

Once the setup is completed, the dependencies can be installed in a new virtual environment.

Open a jupyter notebook and run it.

## Motor Imagery Experiment

Motor imagery experiment incorporating with SPDIM and TSMNet. currently a public EEG BCI datasets are supported as an example: [BNCI2015001](http://bnci-horizon-2020.eu/database/data-sets)
The [moabb](https://neurotechx.github.io/moabb/) and [mne](https://mne.tools) packages are used to download and preprocess these datasets. <br>
**Notice**: there is no need to manually download and preprocess the datasets. This is done automatically on the fly

More detail instructions  are described in the `demo.ipynb` notebook.




