# Agree to Disagree: Robust Anomaly Detection with Noisy Labels
This repository contains the code for our paper *[Agree to Disagree: Robust Anomaly Detection with Noisy Labels](https://dl.acm.org/doi/10.1145/3709657)* which was accepted at SIGMOD 2025.

## Unity Overview
Unity converts the unsupervised anomaly detection problem into a learning-from-noisy-labels problem. Given a dataset X and initial pseudo labels containing label noise (Ŷ), Unity elegantly combines clean sample selection and label refurbishment to iteratively curate a diverse set of clean samples for training. The figure below overviews each component of Unity.

### Unity Architecture:

## Running Unity
The code in this repository was developed using Python 3.9.12 and was ran on an A100 GPU. [`requirements.txt`](./requirements.txt) contains a list of all the required packages and versions. To run our code, run [main.py](./main.py) which takes in a variety of command line arguments to run different methods, datasets, experiments, and modify various parameters. The different arguments and their allowed values are shown below.

| flag         | purpose                                              | allowed values                                                                         |
| ------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `--lr_co_teaching`  | peer networks learning rate                                     | float value                                                                                                      |
| `--lr_contrastive`  | embedding network learning rate                                 | float value                                                                                                      |
| `--hidden_dim`      | hidden dim for peer networks                                    | integer value                                                                                                    |
| `--label_noise`     | initial noisy label generator                                   | `isolation_forest`                                                                                               |
| `--momentum`        | momentum for all optimizers                                     | float value                                                                                                      |
| `--batch_size`      | batch size for training                                         | integer value                                                                                                    |
| `--col_name_labels` | name of column in dataset with GT labels                        | string value                                                                                                     |
| `--w_b`             | run Unity with w&b                                              | `1` for w&b `0` else                                                                                             |
| `--strategy`        | for SOTA methods, controls how sample selection should proceed  | `split` or `default`                                                                                             |
| `--dataset`         | dataset to run                                                  | annthyroid, cardio, landsat, mammography, mnist, pageblocks, pendigits, thyroid, wine, wpbc                      |
| `--method`          | the method to run                                               | `UNITY`, `co_teaching`, `co_teaching_plus`, `jocor`, `clean`, `UADB`, `SELFIE`, `SEAL`, `noisy_label_generator`  |
| `--seed`            | the seed to run                                                 | integer value                                                                                                    |
| `--epochs_warmup`   | the number of epochs to warm up with                            | integer value                                                                                                    |
| `--beta`            | trade-off between historical data and newer data for EMA        | float value between 0 and 1                                                                                      |
| `--alpha`           | trade-off between loss and similarity score or confidence score | float value between 0 and 1                                                                                      |
| `--batch_norm`      | batchnorm layers for the peer networks                          | `true` or `false`                                                                                                |
| `--experiment`      | the experiment to run                                           | `unity`, `disagree_only`, `agree_only`, `agree_and_disagree`, `refurbishment_only`, `varying_noise_rate`, `complexity`, `no_threshold`, `tsne`   |

## Citation
Thank you for your interest in our work, please use the following citation when referencing Unity.

```BibTeX
@article{10.1145/3709657,
author = {Hofmann, Dennis M. and VanNostrand, Peter M. and Ma, Lei and Zhang, Huayi and DeOliveira, Joshua C. and Cao, Lei and Rundensteiner, Elke A.},
title = {Agree to Disagree: Robust Anomaly Detection with Noisy Labels},
year = {2025},
issue_date = {February 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {3},
number = {1},
url = {https://doi.org/10.1145/3709657},
doi = {10.1145/3709657},
journal = {Proc. ACM Manag. Data},
month = feb,
articleno = {7},
numpages = {24},
}
```
