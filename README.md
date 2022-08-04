# DeepDPM: Deep Clustering With An Unknown Number of Clusters
This repo contains the official implementation of our CVPR 2022 paper:
> [**DeepDPM: Deep Clustering With An Unknown Number of Clusters**](https://arxiv.org/abs/2203.14309)
>
> [Meitar Ronen](https://www.linkedin.com/in/meitar-ronen/), [Shahaf Finder](https://shahaffind.github.io) and [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/index.htm).

- [paper \& supp mat](https://arxiv.org/abs/2203.14309).

DeepDPM clustering example on 2D data.<br />
On the left: DeepDPM's predicted clusters' assignments, centers and covariances. On the right: Clusters colored by the GT labels, and the net's decision boundary.
<br>
<p align="center">
<img src="clustering_example.gif" width="750" height="600">
</p>


Examples of the clusters found by DeepDPM on the ImageNet Dataset:


![Examples of the clusters found by DeepDPM on the ImageNet dataset](ImageNet_cluster_examples/cluster_examples.jpg?raw=true "Examples of the clusters found by DeepDPM on the ImageNet dataset")


##### Table of Contents  
1. [Introduction](#Introduction)  
2. [Installation](#Installation)
3. [Training](#Training)
4. [Inference](#Inference)
5. [Citation](#Citation)


## Introduction
DeepDPM is a nonparametric deep-clustering method which unlike most deep clustering methods, does not require knowing the number of clusters, K; rather, it infers it as a part of the overall learning. Using a split/merge framework to change the clusters number adaptively and a novel loss, our proposed method outperforms existing (both classical and deep) nonparametric methods.

While the few existing deep nonparametric methods lack scalability, we show ours by being the first such method that reports its performance on ImageNet.

## Installation
The code runs with Pytorch version 3.9.
Assuming Anaconda, the virtual environment can be installed using:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pytorch-lightning=1.2.10
conda install -c conda-forge umap-learn
conda install -c conda-forge neptune-client
pip install kmeans-pytorch
conda install psutil numpy pandas matplotlib scikit-learn scipy seaborn tqdm joblib
```
See the requirements.txt file for an overview of the packages in the environment we used to produce our results.

## Training
### Setup
#### Datasets and embeddings

When training on raw data (e.g., on MNIST, Reuters10k) the data for MNIST will be automatically downloaded to the "data" directory. For reuters10k, the user needs to download the dataset independently (available online) into the "data" directory.

### Logging
To run the following with logging enabled, edit DeepDPM.py and DeepDPM_alternations.py and insert your neptune token and project path. Alternatively, run the following script with the --offline flag to skip logging. Evaluation metrics will be printed at the end of the training in both cases.


### Training models

We provide two models which can be used for clustering: DeepDPM which clusters embedded data and DeepDPM_alternations which alternates between feature learning using an AE and clustering using DeepDPM. 

1. Key hyperparameters:
  - --gpus specifies the number of GPUs to use. E.g., use "--gpus 0" to use one gpu.
  - --offline runs the model without logging
  - --use_labels_for_eval: run the model with ground truth labels for evaluation (labels are not used in the training process). Do not use this flag if you do not have labels.
  - --dir specifies the directory where the train_data and test_data tensors are expected to be saved
  - --init_k the initial guess for K.
  - --start_computing_params specifies when to start computing the clusters' parameters (the M-step) after initialization. When changing this it is important to see that the network had enough time to learn the initializatiion
  - --split_merge_every_n_epochs specifies the frequency of splits and merges
  - --hidden_dims specifies the AE's hidden dimension layers and depth for DeepDPM_alternations 
  - --latent_dim specifies the AE's learned embeddings dimension (the dimension of the features that would be clustered)

  Please also note the NIIW hyperparameters and the guidelines on how to choose them as described in the supplementary material.

2. Training examples:

  - To generate a similar gif to the one presented above, run:
    python DeepDPM.py --dataset synthetic --log_emb every_n_epochs --log_emb_every 1

  - To run DeepDPM on pretrained embeddings (including custom ones):
    ```
    python DeepDPM.py --dataset <dataset_name> --dir <embeddings path>
    ```
    - for example, for MNIST run:
      ```
      python DeepDPM.py --dataset MNIST --dir "./pretrained_embeddings/umap_embedded_datasets/MNIST"
      ```
    - For the imbalanced case use the data dir accordingly, e.g. for MNIST:
      ```
      python DeepDPM.py --dataset MNIST --dir "./pretrained_embeddings/umap_embedded_datasets/MNIST_IMBALANCED"
      ```

    - To run on STL10: 
    ```
    python DeepDPM.py --dataset stl10 --init_k 3 --dir pretrained_embeddings/MOCO/STL10 --NIW_prior_nu 514 --prior_sigma_scale 0.05
    ```
    (note that for STL10 there is no imbalanced version)

  - DeepDPM with feature extraction pipeline (jointly learning clustering and features):
    - For MNIST run:
    ```
    python DeepDPM_alternations.py --latent_dim 10 --dataset mnist --lambda_ 0.005 --lr 0.002 --init_k 3 --train_cluster_net 200 --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --dir <path_to_dataset_location> --pretrain_path ./saved_models/ae_weights/mnist_e2e.zip --number_of_ae_alternations 3 --transform_input_data None --log_metrics_at_train True
    ```
    - For Reuters10k run:
    ```
    python DeepDPM_alternations.py --dataset reuters10k --dir <path_to_dataset_location> --hidden-dims 500 500 2000 --latent_dim 75 --pretrain_path ./saved_models/ae_weights/reuters10k_e2e.zip --NIW_prior_nu 80 --init_k 1 --lambda_ 0.1 --beta 0.5 --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --number_of_ae_alternations 3 --log_metrics_at_train True --dir ./data/
    ```
    - For ImageNet-50:
    ``` 
    python DeepDPM_alternations.py --latent_dim 10 --lambda_ 0.05 --beta 0.01 --dataset imagenet_50 --init_k 10 --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --dir ./pretrained_embeddings/MOCO/IMAGENET_50/ --NIW_prior_nu 12 --pretrain_path ./saved_models/ae_weights/imagenet_50_e2e.zip --prior_sigma_scale 0.0001 --prior_sigma_choice data_std --number_of_ae_alternations 2
    ```
    - For ImageNet-50 imbalanced:
    ```
    python DeepDPM_alternations.py --latent_dim 10 --lambda_ 0.05 --beta 0.01 --dataset imagenet_50_imb --init_k 10  --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --dir ./pretrained_embeddings/MOCO/IMAGENET_50_IMB/ --NIW_prior_nu 12 --pretrain_path ./saved_models/ae_weights/imagenet_50_imb.zip --prior_sigma_choice data_std --prior_sigma_scale 0.0001 --number_of_ae_alternations 4
    ```

  3. Training on custom datasets:
  DeepDPM is desinged to cluster data in the feature space. 
  For dimensionality reduction, we suggest using UMAP, an Autoencoder, or off-the-shelf unsupervised feature extractors like MoCO, SimCLR, swav, etc.
  If the input data is relatively low dimensional (e.g.  <= 128D), it is possible to train on the raw data.

  To load custom data, create a directory that contains two files: train_data.pt and test_data.pt, a tensor for the train and test data respectively.
  DeepDPM would automatically load them. If you have labels you wish to load for evaluation, please use the --use_labels_for_eval flag.

  Note that the saved models in this repo are per dataset, and in most of the cases specific to it. Thus, it is not recommended to use for custom data.

## Inference
For loading a pretrained model from a saved checkpoint, and for an inference example, see: scripts\DeepDPM_load_from_checkpoint.py

## Citation

For any questions: meitarr@post.bgu.ac.il

Contributions, feature requests, suggestion etc. are welcomed.

If you use this code for your work, please cite the following:

```
@inproceedings{Ronen:CVPR:2022:DeepDPM,
  title={DeepDPM: Deep Clustering With An Unknown Number of Clusters},
  author={Ronen, Meitar and Finder, Shahaf E. and  Freifeld, Oren},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```