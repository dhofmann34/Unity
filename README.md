# Agree to Disagree: Robust Anomaly Detection with Noisy Labels
This repository contains the code for our paper Agree to Disagree: Robust Anomaly Detection with Noisy Labels. 

## Running our method Unity
The code in this repository was developed using Python 3.9.12. [`requirements.txt`](./requirements.txt) contains a list of all the required packages and versions. To run our code run [main.py](./main.py) which takes in a variety of command line arguments to run different methods, experiments, and modify various parameters. The different arguments and their allowed values are shown below.

| flag         | purpose                                              | allowed values                                                                         |
| ------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `--lr_co_teaching`  | peer networks learning rate                             | `simple`, `ntrees`, `nrects`, `compare`, `k`, `m`, `nconstraints`, `perturb`, `robust` |
| `--lr_contrastive`  | embedding network learning rate                     | space separated list of values e.g. `10 50 100` or `0.1 0.2 0.3`                       |
| `--hidden_dim`      | hidden dim for peer networks                               | cancer, glass, magic, spambase, vertebral                                              |
| `--label_noise`     | initial noisy label generator                                | `FACETIndex`, `OCEAN`, `RFOCSE`, `AFT`, `MACE`                                         |
| `--momentum`        | momentum for all optimizers                             | integer value, overridden in for `--expr` `ntrees`                                     |
| `--batch_size`      | batch size for training                      | integer value, `-1` for no max depth                                                   |
| `--col_name_labels` | the iteration to run, used as random seed            | space separated integer values                                                         |
| `--w_b`             | a filename modifier append to append to results file | string value                                                                           |
| `--strategy`        | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--dataset`         | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--method`          | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--seed`            | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--epochs_warmup`   | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--beta`            | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--alpha`           | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--batch_norm`      | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
| `--experiment`      | the underlying mode to explain                       | `rf` or `gbc`                                                                          |
