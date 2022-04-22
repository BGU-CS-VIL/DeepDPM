# DeepDPM: Deep Clustering With An Unknown Number of Clusters
This repo contains the implementation of our paper:
> [**DeepDPM: Deep Clustering With An Unknown Number of Clusters**](https://arxiv.org/abs/2203.14309)
>
> [Meitar Ronen](https://www.linkedin.com/in/meitar-ronen/), [Shahaf Finder](https://shahaffind.github.io) and [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/index.htm).

- Accepted at CVPR 2022 ([paper \& supp mat](https://arxiv.org/abs/2203.14309)).

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


### Train model
** In all the modules, use the --gpus to specify the number of GPUs to use.
E.g., to run on one GPU add "--gpus 0" to the command. To run on CPU simply omit the --gpus flag.

1. To generate a similar gif to the one presented above, run:
python DeepDPM.py --dataset synthetic --latent_dim 2 --log_emb every_n_epochs --log_emb_every 1

2. DeepDPM on pretrained embeddings:
```
python DeepDPM.py --dataset <dataset_name>
```
<dataset_name> options: [MNIST_N2D, USPS_N2D, FASHION_N2D]
for example, for MNIST run:
```
python DeepDPM.py --dataset MNIST_N2D
```
- For the imbalanced case run:
```
python DeepDPM.py --dataset <dataset_name> --imbalanced
```

- To run on STL10: 
```
python DeepDPM.py --dataset stl10 --init_k 3 --dir pretrained_embeddings/MOCO/ --latent_dim 512 --prior_nu 514 --prior_sigma_scale 0.05
```
(note that for STL10 there is no imbalanced version)

2. DeepDPM with feature extraction pipeline (jointly learning clustering and features):
- For MNIST run:
```
python DeepDPM_alternations.py --latent_dim 10 --dataset mnist --lambda_ 0.005 --lr 0.002 --init_k 3 --train_cluster_net 200 --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --dir <path_to_datasets_location> --pretrain_path ./saved_models/ae_weights/mnist_e2e.zip --number_of_ae_alternations 3 --transform None --log_metrics_at_train True
```


- For Reuters10k run:
```
python DeepDPM_alternations.py --dataset reuters10k --dir <path_to_datasets_location> --hidden-dims 500 500 2000 --latent_dim 75 --pretrain_path ./saved_models/ae_weights/reuters10k_e2e.zip --prior_nu 80 --init_k 1 --lambda_ 0.1 --beta 0.5 --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --number_of_ae_alternations 3 --log_metrics_at_train True --dir ./data/
```
- For ImageNet-50:
```
python DeepDPM_alternations.py --latent_dim 10 --lambda_ 0.05 --beta 0.01 --dataset imagenet_50 --init_k 10 --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --dir ./pretrained_embeddings/MOCO/ --prior_nu 12 --pretrain_path ./saved_models/ae_weights/imagenet_50_e2e.zip --prior_sigma_scale 0.0001 --prior_sigma_choice data_std --number_of_ae_alternations 2
```
- For ImageNet-50 imbalanced:
```
python DeepDPM_alternations.py --latent_dim 10 --lambda_ 0.05 --beta 0.01 --dataset imagenet_50_imb --init_k 10  --alternate --init_cluster_net_using_centers --reinit_net_at_alternation --dir ./pretrained_embeddings/MOCO/ --prior_nu 12 --pretrain_path ./saved_models/ae_weights/imagenet_50_imb --prior_sigma_choice data_std --prior_sigma_scale 0.0001 --number_of_ae_alternations 4
```

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

