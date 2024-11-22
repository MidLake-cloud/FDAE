# FDAE

The pytorch implementation of our model FDAE. Some codes about data preprocessing and SVM refer to https://github.com/mondejar/ecg-classification.

## Preparation

1. Prepare a virtual environment with Python 3.7.
2. run `pip install -r requirements.txt` to install extra packages.
3. Download MITBIH dataset from https://www.kaggle.com/datasets/mondejar/mitbih-database?resource=download and put it has in the folder `dataset/archive/mitbih_database/`.

## Get Started

Run

`python utils/make_pair_dataset.py`

to make the dataset for contrastive learning.

Then run

`python experiments/run_disnet.py`

to train and generate ECGs.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:
- https://github.com/mondejar/ecg-classification
- https://github.com/DreamStudioAI/sim_gan