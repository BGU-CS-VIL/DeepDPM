import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def check_args(args, data_dim):
    # Make sure that the NIW prior's nu is defined correctly
    args.NIW_prior_nu = args.NIW_prior_nu or (data_dim + 2)
    if args.NIW_prior_nu < data_dim + 1:
        raise Exception(f"The chosen NIW nu hyperparameter need to be at least D+1 (D is the data dim). Set --NIW_prior_nu to at least {data_dim + 1}")

    # Ensure that there is no evaluation if no labels exist
    if not args.use_labels_for_eval:
        args.evaluate_every_n_epochs = 0

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(row_ind)):
            if row_ind[j] == y_pred[i]:
                best_fit.append(col_ind[j])
    return best_fit, row_ind, col_ind, w

def cluster_acc(y_true, y_pred):
    best_fit, row_ind, col_ind, w = best_cluster_fit(y_true, y_pred)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size