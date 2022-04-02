#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
from kmeans_pytorch import kmeans as GPU_KMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def init_mus_and_covs(codes, K, how_to_init_mu, logits, use_priors=True, prior=None, random_state=0, device="cpu"):
    """This function initalizes the clusters' centers and covariances matrices.

    Args:
        codes (torch.tensor): The codes that should be clustered, in R^{N x D}.
        how_to_init_mu (str): A string defining how to initialize the centers.
        use_priors (bool, optional): Whether to consider the priors. Defaults to True.
    """
    print("Initializing clusters params using Kmeans...")
    if codes.shape[0] > 2 * (10 ** 5):
        # sample only a portion of the codes
        codes = codes[:2 * (10**5)]
    if how_to_init_mu == "kmeans":
        if K == 1:
            kmeans = KMeans(n_clusters=K, random_state=random_state).fit(codes.detach().cpu())
            labels = torch.from_numpy(kmeans.labels_)
            kmeans_mus = torch.from_numpy(kmeans.cluster_centers_)
        else:
            labels, kmeans_mus = GPU_KMeans(X=codes.detach(), num_clusters=K, device=device)
        _, counts = torch.unique(labels, return_counts=True)
        pi = counts / float(len(codes))
        data_covs = compute_data_covs_hard_assignment(labels, codes, K, kmeans_mus.cpu(), prior)

        if use_priors:
            mus = prior.compute_post_mus(counts, kmeans_mus.cpu())
            covs = []
            for k in range(K):
                codes_k = codes[labels == k]
                cov_k = prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])  # len(codes_k) == counts[k]? yes
                covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            mus = kmeans_mus
            covs = data_covs
        return mus, covs, pi, labels

    elif how_to_init_mu == "kmeans_1d":
        pca = PCA(n_components=1)
        pca_codes = pca.fit_transform(codes.detach().cpu())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=K, device=torch.device(device))

        kmeans_mus = torch.tensor(
            pca.inverse_transform(cluster_centers.cpu().numpy()),
            device=device,
            requires_grad=False,
        )
        _, counts = torch.unique(torch.tensor(labels), return_counts=True)
        pi = counts / float(len(codes))
        data_covs = compute_data_covs_hard_assignment(labels, codes, K, kmeans_mus.cpu(), prior)

        if use_priors:
            mus = prior.compute_post_mus(counts, kmeans_mus.cpu())
            covs = []
            for k in range(K):
                codes_k = codes[labels == k]
                cov_k = prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])  # len(codes_k) == counts[k]? yes
                covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            mus = kmeans_mus
            covs = data_covs

        return mus, covs, pi, labels

    elif how_to_init_mu == "soft_assign":
        mus = compute_mus_soft_assignment(codes, logits, K)
        pi = compute_pi_k(logits, prior=prior if use_priors else None)
        data_covs = compute_data_covs_soft_assignment(logits, codes, K, mus.cpu(), prior.name)

        if use_priors:
            mus = prior.compute_post_mus(pi, mus)
            covs = []
            for k in range(K):
                r_k = pi[k] * len(codes)  # if it the sum of logits change to this becuase this is confusing
                cov_k = prior.compute_post_cov(r_k, mus[k], data_covs[k])
                covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            covs = data_covs
        return mus, covs, pi, logits.argmax(axis=1)


def init_mus_and_covs_sub(codes, k, n_sub, how_to_init_mu_sub, logits, logits_sub, prior=None, use_priors=True, random_state=0, device="cpu"):
    if how_to_init_mu_sub == "kmeans":
        counts = []
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        if len(codes_k) <= n_sub:
            # empty cluster
            codes_k = codes
        
        labels, cluster_centers = GPU_KMeans(X=codes_k.detach(), num_clusters=n_sub, device=torch.device('cuda:0'))

        if len(codes[indices_k]) <= n_sub:
            c = torch.tensor([0, len(codes[indices_k])])
        else:
            _, c = torch.unique(labels, return_counts=True)
        counts.append(c)
        mus_sub = cluster_centers

        data_covs_sub = compute_data_covs_hard_assignment(labels, codes_k, n_sub, mus_sub, prior)
        if use_priors:
            mus_sub = prior.compute_post_mus(counts, mus_sub.cpu())
            covs_sub = []
            for k in range(n_sub):
                covs_sub_k = prior.compute_post_cov(counts[k], codes_k[labels == k].mean(axis=0), data_covs_sub[k])
                covs_sub.append(covs_sub_k)
            covs_sub = torch.stack(covs_sub)
        else:
            covs_sub = data_covs_sub

        pi_sub = torch.cat(counts) / float(len(codes))
        return mus_sub, covs_sub, pi_sub

    elif how_to_init_mu_sub == "kmeans_1d":
        # pca codes to 1D then perform 1d kmeans
        counts = []
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        if len(codes_k) <= n_sub:
            # empty cluster
            codes_k = codes
        pca = PCA(n_components=1).fit(codes_k.detach().cpu())
        pca_codes = pca.fit_transform(codes_k.detach().cpu())
        # kmeans = KMeans(n_clusters=n_sub, random_state=random_state).fit(pca_codes)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=n_sub, device=torch.device(device))

        if len(codes[indices_k]) <= n_sub:
            c = torch.tensor([0, len(codes[indices_k])])
        else:
            _, c = torch.unique(labels, return_counts=True)
        counts.append(c)
        counts = counts[0]

        mus_sub = torch.tensor(
            pca.inverse_transform(cluster_centers.cpu().numpy()),
            device=device,
            requires_grad=False,
        )

        data_covs_sub = compute_data_covs_hard_assignment(labels, codes_k, n_sub, mus_sub, prior)
        if use_priors:
            mus_sub = prior.compute_post_mus(counts, mus_sub.cpu())
            covs_sub = []
            for k in range(n_sub):
                cov_sub_k = prior.compute_post_cov(counts[k], codes_k[labels == k].mean(axis=0), data_covs_sub[k])
                covs_sub.append(cov_sub_k)
            covs_sub = torch.stack(covs_sub)
        else:
            covs_sub = data_covs_sub

        pi_sub = counts / float(len(codes))
        return mus_sub, covs_sub, pi_sub

    elif how_to_init_mu_sub == "soft_assign":
        raise NotImplementedError()


def compute_data_sigma_sq_hard_assignment(labels, codes, K, mus):
    # returns K X D
    sigmas_sq = []
    for k in range(K):
        codes_k = codes[labels == k]
        sigmas_sq.append(codes_k.std(axis=0) ** 2)
    return torch.stack(sigmas_sq)


def compute_data_covs_hard_assignment(labels, codes, K, mus, prior):
    if prior and prior.mus_covs_prior.name == "NIG":
        return compute_data_sigma_sq_hard_assignment(labels, codes, K, mus)
    else:
        covs = []
        for k in range(K):
            codes_k = codes[labels == k]
            N_k = float(len(codes_k))
            if N_k > 0:
                cov_k = torch.matmul(
                    (codes_k - mus[k].cpu().repeat(len(codes_k), 1)).T,
                    (codes_k - mus[k].cpu().repeat(len(codes_k), 1)),
                )
                cov_k = cov_k / N_k
            else:
                if prior:
                    _, cov_k = prior.init_priors()
                else:
                    cov_k = torch.eye(codes.shape[1]) * 0.0005
            covs.append(cov_k)
        return torch.stack(covs)


def compute_data_sigma_sq_soft_assignment(codes, logits, K, mus):
    # Assuming the mus were also computed using soft assignments (mu is a weighted sample mean)

    denominator = logits.sum(axis=0)  # sum over all points per K
    stds = torch.stack([
        (logits[:, k].unsqueeze(1) * ((codes - mus[k])**2)).sum(axis=0) / denominator[k]
        for k in range(K)
    ])
    return stds


def compute_mus_soft_assignment(codes, logits, K, constant=True):
    # gives the embeddings (codes) and their probabilities to be sampled from the K classes, return each cluster's mu.
    # soft_assign (logits) is [N_batch X K], codes are [N_batch X feat_dim]
    denominator = logits.sum(axis=0)  # sum over all points per K
    # for each k, we are multiplying the k-th column of r with the codes matrix element-wise (first element * first row of c,...).
    # then, we are summing over all the data points (over the rows) and dividing by the normalizer
    # finally we are stacking all the mus.
    mus = torch.stack(
        [
            (logits[:, k].reshape(-1, 1) * codes).sum(axis=0) / denominator[k]
            for k in range(K)
        ]
    )  # K x feat_dim
    if constant:
        mus = mus.detach()
    return mus


def compute_pi_k(logits, prior=None):
    N = logits.shape[0]
    # sum for prob for each K (across all points) \sum_{i=1}^{N}P(z_i = k)
    r_sum = logits.sum(dim=0)
    if len(r_sum.shape) > 1:
        # this is sub clusters' pi need another sum
        r_sum = r_sum.sum(axis=0)
    pi = r_sum / torch.tensor(N, dtype=torch.float64)
    if prior:
        pi = prior.comp_post_pi(pi)
    return pi


def compute_data_covs_soft_assignment(logits, codes, K, mus, prior_name="NIW"):
    # compute the data covs in soft assignment
    prior_name = prior_name or "NIW"
    if prior_name == "NIW":
        covs = []
        n_k = logits.sum(axis=0)
        n_k += 0.0001
        for k in range(K):
            if len(logits) == 0 or len(codes) == 0:
                # happens when finding subcovs of empty clusters
                cov_k = torch.eye(mus.shape[1]) * 0.0001
            else:
                cov_k = torch.matmul(
                    (logits[:, k] * (codes - mus[k].repeat(len(codes), 1)).T),
                    (codes - mus[k].repeat(len(codes), 1)),
                )
                cov_k = cov_k / n_k[k]
            covs.append(cov_k)
        return torch.stack(covs)
    elif prior_name == "NIG":
        return compute_data_sigma_sq_soft_assignment(logits=logits, codes=codes, K=K, mus=mus)


def compute_mus(codes, logits, pi, K, how_to_compute_mu, use_priors=True, prior=None, random_state=0, device="cpu"):
    if how_to_compute_mu == "kmeans":
        labels, cluster_centers = GPU_KMeans(X=codes.detach(), num_clusters=K, device=torch.device('cuda:0'))
        mus = cluster_centers
    elif how_to_compute_mu == "soft_assign":
        mus = compute_mus_soft_assignment(codes, logits, K)

    if use_priors:
        counts = pi * len(codes)
        mus = prior.compute_post_mus(counts, mus)
    else:
        mus = mus
    return mus


def compute_covs(codes, logits, K, mus, use_priors=True, prior=None):
    data_covs = compute_data_covs_soft_assignment(codes=codes, logits=logits, K=K, mus=mus, prior_name=prior.name if prior else None)
    if use_priors:
        covs = []
        r = logits.sum(axis=0)
        for k in range(K):
            cov_k = prior.compute_post_cov(r[k], mus[k], data_covs[k])
            covs.append(cov_k)
        covs = torch.stack(covs)
    else:
        covs = torch.stack([torch.eye(mus.shape[1]) * data_covs[k] for k in range(K)])
    return covs


def compute_mus_covs_pis_subclusters(codes, logits, logits_sub, mus_sub, K, n_sub, hard_assignment=True, use_priors=True, prior=None):
    pi_sub = compute_pi_k(logits_sub, prior=prior if use_priors else None)
    if hard_assignment:
        mus_sub_new, covs_sub_new = [], []
        for k in range(K):
            indices = logits.argmax(-1) == k
            codes_k = codes[indices]
            r_sub = logits_sub[indices, 2 * k: 2 * k + 2]
            denominator = r_sub.sum(axis=0)  # sum over all points per K

            if indices.sum() < 2 or denominator[0] == 0 or denominator[1] == 0 or len(torch.unique(r_sub.argmax(-1))) < n_sub:
                # Empty subcluster encountered, re-initializing cluster {k}
                mus_sub, covs_sub, pi_sub_ = init_mus_and_covs_sub(codes=codes, k=k, n_sub=n_sub, logits=logits, logits_sub=logits_sub, how_to_init_mu_sub="kmeans_1d", prior=prior, use_priors=use_priors, device=codes.device)
                pi_sub[2*k: 2*k+2] = pi_sub_
                mus_sub_new.append(mus_sub[0])
                mus_sub_new.append(mus_sub[1])
                covs_sub_new.append(covs_sub[0])
                covs_sub_new.append(covs_sub[1])
            else:
                mus_sub_k = []
                for k_sub in range(n_sub):
                    z_sub = r_sub[:, k_sub]
                    mus_sub_k.append(
                        (z_sub.reshape(-1, 1) * codes_k.cpu()).sum(axis=0)
                        / denominator[k_sub]
                    )
                mus_sub_new.extend(mus_sub_k)
                data_covs_k = compute_data_covs_soft_assignment(r_sub, codes_k, n_sub, mus_sub_k, prior.name)
                if use_priors:
                    covs_k = []
                    for k_sub in range(n_sub):
                        cov_k = data_covs_k[k_sub]
                        if torch.isnan(cov_k).any():
                            # at least one of the subclusters has empty assignments
                            cov_k = torch.eye(cov_k.shape[0]) * prior.mus_covs_prior.prior_sigma_scale  # covs_sub[2 * k]
                        cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], mus_sub_k[k_sub], cov_k)
                        covs_k.append(cov_k)
                else:
                    covs_k = data_covs_k
                covs_sub_new.extend(covs_k)
        mus_sub_new = torch.stack(mus_sub_new)
    if use_priors:
        counts = pi_sub * len(codes)
        mus_sub_new = prior.compute_post_mus(counts, mus_sub_new)
    covs_sub_new = torch.stack(covs_sub_new)
    return mus_sub_new, covs_sub_new, pi_sub


def compute_mus_subclusters(codes, logits, logits_sub, pi_sub, mus_sub, K, n_sub, hard_assignment=True, use_priors=True, prior=None):
    if hard_assignment:
        # Data term
        mus_sub_new = []
        for k in range(K):
            denominator = logits_sub[:, 2 * k: 2 * k + 2].sum(
                    axis=0
                )  # sum over all points per K
            indices = logits.argmax(-1) == k
            if indices.sum() < 5:
                # empty cluster - do not change mu sub
                mus_sub_new.append(
                    mus_sub[2 * k: 2 * k + 2].clone().detach().cpu().type(torch.float32)
                )
            else:
                codes_k = codes[indices]
                for k_sub in range(n_sub):
                    if denominator[k_sub] == 0:
                        # empty cluster - do not change mu sub
                        mus_sub_new.append(
                            mus_sub[2 * k + k_sub].clone().detach().cpu().type(torch.float32).unsqueeze(0)
                        )
                    else:
                        z_sub = logits_sub[indices, 2 * k + k_sub]

                        mus_sub_new.append(
                            ((z_sub.reshape(-1, 1) * codes_k.cpu()).sum(axis=0)
                             / denominator[k_sub]).unsqueeze(0)
                        )
    mus_sub_new = torch.cat(mus_sub_new)

    if use_priors and prior:
        counts = pi_sub * len(codes)
        mus_sub_new = prior.compute_post_mus(counts, mus_sub_new)
    return mus_sub_new


def compute_covs_subclusters(codes, logits, logits_sub, K, n_sub, mus_sub, covs_sub, pi_sub, use_priors=True, prior=None):
    for k in range(K):
        indices = logits.argmax(-1) == k
        codes_k = codes[indices]
        r_sub = logits_sub[indices, 2 * k: 2 * k + 2]
        data_covs_k = compute_data_covs_soft_assignment(r_sub, codes_k, n_sub, mus_sub[2 * k: 2 * k + 2], prior.name)
        if use_priors:
            covs_k = []
            for k_sub in range(n_sub):
                cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], mus_sub[2 * k + k_sub], data_covs_k[k_sub])
                covs_k.append(cov_k)
            covs_k = torch.stack(covs_k)
        else:
            covs_k = data_covs_k
        if torch.isnan(cov_k).any():
            # at least one of the subclusters has empty assignments
            if torch.isnan(cov_k[0]).any():
                # first subcluster is empty give last cov
                covs_k[0] = covs_sub[2 * k]
            if torch.isnan(cov_k[1]).any():
                covs_k[1] = covs_sub[2 * k + 1]
        if k == 0:
            covs_sub_new = covs_k
        else:
            covs_sub_new = torch.cat([covs_sub_new, covs_k])
    return covs_sub_new


def _create_subclusters(k_sub, codes, logits, logits_sub, mus_sub, pi_sub, n_sub, how_to_init_mu_sub, prior, device=None, random_state=0, use_priors=True):
    # k_sub is the index of sub mus that now turns into a mu
    # Recieves as input a vector of mus and generates two subclusters of it
    device= device or codes.device
    D = mus_sub.shape[1]
    if how_to_init_mu_sub == "soft_assign":
        mu_1 = (
            mus_sub[k_sub]
            + mus_sub[k_sub] @ torch.eye(D, D) * 0.05
        )
        mu_2 = (
            mus_sub[k_sub]
            - mus_sub[k_sub] @ torch.eye(D, D) * 0.05
        )
        new_covs = torch.stack([0.05 for i in range(2)])
        new_pis = torch.tensor([0.5, 0.5]) * pi_sub[k_sub]
        new_mus = torch.stack([mu_1, mu_2]).squeeze(dim=1)
        use_priors = False
        # return mus, covs, pis
    elif how_to_init_mu_sub == "kmeans" or "kmeans_1d":
        indices_k = logits.argmax(-1) == k_sub // 2
        codes_k = codes[indices_k, :]
        if len(logits_sub) > 0:
            sub_assignment = logits_sub.argmax(-1)
            codes_sub = codes[sub_assignment == k_sub]
        else:
            # comp assignments by min dist
            k_sub_other = k_sub + 1 if k_sub % 2 == 0 else k_sub - 1
            sub_assignment = comp_subclusters_params_min_dist(codes_k, mus_sub[k_sub], mus_sub[k_sub_other])
            codes_sub = codes_k[sub_assignment == (k_sub % 2)]  # sub_assignment is in range 0 and 1.

        if how_to_init_mu_sub == "kmeans":
            labels, cluster_centers = GPU_KMeans(X=codes_sub.detach(), num_clusters=n_sub, device=torch.device('cuda:0'))
            new_mus = cluster_centers.cpu()
            new_covs = compute_data_covs_hard_assignment(labels=labels, codes=codes_sub, K=n_sub, mus=new_mus, prior=prior)
            _, new_pis = torch.unique(
                labels, return_counts=True
            )
            new_pis = (new_pis / float(len(codes_sub))) * pi_sub[k_sub]
        elif how_to_init_mu_sub == "kmeans_1d":
            # kmeans_1d
            pca = PCA(n_components=1).fit(codes_sub.detach().cpu())
            pca_codes = pca.fit_transform(codes_sub.detach().cpu())

            device = "cuda" if torch.cuda.is_available() else "cpu"
            labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=n_sub, device=torch.device(device))

            new_mus = torch.tensor(
                pca.inverse_transform(cluster_centers.cpu().numpy()),
                device=device,
                requires_grad=False,
            ).cpu()
            new_covs = compute_data_covs_hard_assignment(
                labels=labels, codes=codes_sub, K=n_sub, mus=new_mus, prior=prior
            )
            _, new_pis = torch.unique(
                labels, return_counts=True
            )
            new_pis = (new_pis / float(len(codes_sub))) * pi_sub[k_sub]

        if use_priors:
            _, counts = torch.unique(labels, return_counts=True)
            new_mus = prior.compute_post_mus(counts, new_mus)  # up until now we didn't use this
            covs = []
            for k in range(n_sub):
                new_cov_k = prior.compute_post_cov(counts[k], codes_sub[labels == k].mean(axis=0), new_covs[k])
                covs.append(new_cov_k)
            new_covs = torch.stack(covs)
            pis_post = prior.comp_post_pi(new_pis)  # sum to 1
            new_pis = pis_post * pi_sub[k_sub]  # sum to pi_sub[k_sub]

    return new_mus, new_covs, new_pis


def comp_subclusters_params_min_dist(codes_k, mu_sub_1, mu_sub_2):
    """
    Comp assignments to subclusters by min dist to subclusters centers
    codes_k (torch.tensor): the datapoints assigned to the k-th cluster
    mu_sub_1, mu_sub_2 (torch.tensor, torch.tensor): the centroids of the first and second subclusters of cluster k

    Returns the (hard) assignments vector (in range 0 and 1).
    can be used for e.g.,
    codes_k_1 = codes_k[assignments == 0]
    codes_k_2 = codes_k[assignments == 1]
    """

    dists_0 = torch.sqrt(torch.sum((codes_k - mu_sub_1) ** 2, axis=1))
    dists_1 = torch.sqrt(torch.sum((codes_k - mu_sub_2) ** 2, axis=1))
    assignments = torch.stack([dists_0, dists_1]).argmin(0)
    return assignments
