# Agree to Disagree: Robust Anomaly Detection with Noisy Labels
This repository contains the code for our paper Agree to Disagree: Robust Anomaly Detection with Noisy Labels. 

## Running our method Unity
The code in this repository was developed using Python 3.9.12. [`requirements.txt`](./requirements.txt) contains a list of all the required packages and versions. To run our code run [main.py](./main.py) which takes in a variety of command line arguments to run different methods, experiments, and modify various parameters. The different arguments and their allowed values are shown below.

| flag         | purpose                                              | allowed values                                                                         |
| ------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `--lr_co_teaching`  | peer networks learning rate                                     | `simple`, `ntrees`, `nrects`, `compare`, `k`, `m`, `nconstraints`, `perturb`, `robust` |
| `--lr_contrastive`  | embedding network learning rate                                 | space separated list of values e.g. `10 50 100` or `0.1 0.2 0.3`                       |
| `--hidden_dim`      | hidden dim for peer networks                                    | cancer, glass, magic, spambase, vertebral                                              |
| `--label_noise`     | initial noisy label generator                                   | `FACETIndex`, `OCEAN`, `RFOCSE`, `AFT`, `MACE`                                         |
| `--momentum`        | momentum for all optimizers                                     | integer value, overridden in for `--expr` `ntrees`                                     |
| `--batch_size`      | batch size for training                                         | integer value, `-1` for no max depth                                                   |
| `--col_name_labels` | name of column in dataset with GT labels                        | space separated integer values                                                         |
| `--w_b`             | run Unity with w&b                                              | string value                                                                           |
| `--strategy`        | for SOTA methods, controls how sample selection should proceed  | `rf` or `gbc`                                                                          |
| `--dataset`         | dataset to run                                                  | `rf` or `gbc`                                                                          |
| `--method`          | the method to run                                               | `rf` or `gbc`                                                                          |
| `--seed`            | the seed to run                                                 | `rf` or `gbc`                                                                          |
| `--epochs_warmup`   | the number of epochs to warm up with                            | `rf` or `gbc`                                                                          |
| `--beta`            | trade-off between historical data and newer data for EMA        | `rf` or `gbc`                                                                          |
| `--alpha`           | trade-off between loss and similarity score or confidence score | `rf` or `gbc`                                                                          |
| `--batch_norm`      | batchnorm layers for the peer networks                          | `rf` or `gbc`                                                                          |
| `--experiment`      | the experiment to run                                           | `rf` or `gbc`                                                                          |
