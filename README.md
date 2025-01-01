# Agree to Disagree: Robust Anomaly Detection with Noisy Labels
This repository contains the code for our paper Agree to Disagree: Robust Anomaly Detection with Noisy Labels. 

## Running Unity
The code in this repository was developed using Python 3.9.12. [`requirements.txt`](./requirements.txt) contains a list of all the required packages and versions. To run our code run [main.py](./main.py) which takes in a variety of command line arguments to run different methods, experiments, and modify various parameters. The different arguments and their allowed values are shown below.

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
