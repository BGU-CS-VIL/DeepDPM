import argparse
import torch
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel
from src.embbeded_datasets import embbededDataset

# LOAD MODEL FROM CHECKPOINT
cp_path = CHECKPOINT_PATH # E.g.: "./saved_models/MNIST_N2D/default_exp/epoch=57-step=31725.ckpt"
cp_state = torch.load(cp_path)
K = cp_state['state_dict']['cluster_net.class_fc2.weight'].shape[0] 
hyper_param = cp_state['hyper_parameters']
args = argparse.Namespace()
for key, value in hyper_param.items():
    setattr(args, key, value)

model = ClusterNetModel.load_from_checkpoint(
    checkpoint_path=cp_path,
    input_dim=args.latent_dim,
    init_k = K,
    hparams=args
    )

# Example for inference :
model.eval()
dataset_obj = embbededDataset(args)
_, val_loader = dataset_obj.get_loaders()
cluster_assignments = []
for data, label in val_loader:
    soft_assign = model(data)
    hard_assign = soft_assign.argmax(-1)
    cluster_assignments.append(hard_assign)
print(cluster_assignments)


