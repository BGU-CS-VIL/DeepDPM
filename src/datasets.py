#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

class MyDataset:
    @property
    def input_dim(self):
        return self._input_dim

    def __init__(self, args):
        self.ds_name = args.dataset
        self.args = args
        self.data_dir = os.path.join(args.dir, self.ds_name)

    def get_train_data(self):
        raise NotImplementedError()

    def get_test_data(self):
        raise NotImplementedError()

    def get_train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            self.get_train_data(),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=6,
        )
        return train_loader

    def get_test_loader(self):
        test_loader = torch.utils.data.DataLoader(self.get_test_data(), batch_size=self.args.batch_size, shuffle=False, num_workers=6)
        return test_loader

    def get_loaders(self):
        return self.get_train_loader(), self.get_test_loader()


class MNIST(MyDataset):
    def __init__(self, args):
        super().__init__(args)
        self.transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self._input_dim = 28 * 28

    def get_train_data(self):
        return datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transformer)

    def get_test_data(self):
        return datasets.MNIST(self.data_dir, train=False, transform=self.transformer)
class STL10(MyDataset):
    def __init__(self, args, split="train"):
        super().__init__(args)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self._input_dim = 96 * 96 * 3
        self.data_dir = os.path.join(args.dir, "STL10")
        self.split = split

    def get_train_data(self):
        return datasets.STL10(self.data_dir, split=self.split, download=True, transform=self.transformer)

    def get_test_data(self):
        return datasets.STL10(self.data_dir, split="test", transform=self.transformer)


class USPS(MyDataset):
    """
    https://github.com/nairouz/DynAE/blob/master/DynAE/datasets.py
    """
    def __init__(self, args):
        super().__init__(args)
        self.transformer = transforms.Compose(
            [transforms.ToTensor()]
            # , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self._input_dim = 16 * 16
        self.data_dir = os.path.join(args.dir, "USPS")

    def get_train_data(self):
        if not os.path.exists(self.data_dir + "/usps_train.jf"):
            if not os.path.exists(self.data_dir + "/usps_train.jf.gz"):
                url_train = "http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz"
                url_test = "http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz"
                os.system(f"wget {url_train} -P %s" % self.data_dir)
                os.system(f"wget {url_test} -P %s" % self.data_dir)
            os.system("gunzip %s/usps_train.jf.gz" % self.data_dir)
            os.system("gunzip %s/usps_test.jf.gz" % self.data_dir)

        with open(self.data_dir + "/usps_train.jf") as f:
            data = f.readlines()
        data = data[1:-1]
        data = [list(map(float, line.split())) for line in data]
        data = torch.Tensor(data)

        imgs = data[:, 1:]
        labels = data[:, 0]

        if self.args.transform:
            if self.args.transform == "min_max" or "USPS_N2D" in self.args.dataset:
                data = torch.Tensor(MinMaxScaler().fit_transform(imgs.numpy()))
            elif self.args.transform == "normalize":
                data = torch.Tensor(Normalizer().fit_transform(imgs.numpy()))
            elif self.args.transform == "standard":
                data = torch.Tensor(StandardScaler().fit_transform(imgs.numpy()))
        train_set = TensorDataset(imgs, labels)
        del data, imgs, labels
        return train_set

    def get_test_data(self):
        with open(self.data_dir + "/usps_test.jf") as f:
            data = f.readlines()
        data = data[1:-1]
        data = [list(map(float, line.split())) for line in data]
        data = torch.Tensor(data)
        imgs = data[:, 1:]
        labels = data[:, 0]
        if self.args.transform:
            if self.args.transform == "normalize":
                data = torch.Tensor(Normalizer().fit_transform(imgs.numpy()))
            elif self.args.transform == "min_max":
                data = torch.Tensor(MinMaxScaler().fit_transform(imgs.numpy()))
            elif self.args.transform == "standard":
                data = torch.Tensor(StandardScaler().fit_transform(imgs.numpy()))
        test_set = TensorDataset(imgs, labels)
        del data, imgs, labels
        return test_set

class REUTERS(MyDataset):
    """
    code adapted from
    https://github.com/nairouz/DynAE/blob/master/DynAE/datasets.py
    """
    def __init__(self, args, how_many=None):
        super().__init__(args)
        self.transformer = transforms.Compose(
            [transforms.ToTensor()]
            # , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self._input_dim = 2000
        self.how_many = how_many  # How many samples of REUTERS (e.g., 10K), if None it takes all the dataset
        if how_many is None:
            name = 'reuters'
        elif how_many == 10000:
            name = 'reuters10k'
        else:
            name = f'reuters_{how_many}'
        self.filename = name

    def get_train_data(self):
        data_dir = os.path.join(self.args.dir, "REUTERS")
        if not os.path.exists(os.path.join(data_dir, f"{self.filename}.npy")):
            print("making reuters idf features")
            self.make_reuters_data(data_dir, self.how_many)
            print((f"{self.filename} saved to " + data_dir))

        # data = np.load(os.path.join(data_dir, f"{self.filename}.npy"), allow_pickle=True).item()
        import pickle
        infile = open(os.path.join(data_dir, f"{self.filename}.npy"), 'rb')
        data = pickle.load(infile)
        infile.close()
        # has been shuffled
        x = data["data"]
        y = data["label"]
        x = x.reshape((x.shape[0], -1))
        y = y.reshape((y.size,))
        train_set = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
        del x, y
        return train_set

    def get_test_data(self):
        if self.how_many == 10000:
            return self.get_train_data()
        else:
            # randomly sample
            data_dir = os.path.join(self.args.dir, "REUTERS")
            data = np.load(os.path.join(data_dir, f"{self.filename}.npy"), allow_pickle=True)
            # has been shuffled
            x = data["data"]
            y = data["label"]
            x = x.reshape((x.shape[0], -1))
            y = y.reshape((y.size,))
            test_set = TensorDataset(torch.from_numpy(x)[:8000].float(), torch.from_numpy(y)[:8000].float())
            del x, y
            return test_set

    def make_reuters_data(self, data_dir, how_many):
        np.random.seed(1234)
        from sklearn.feature_extraction.text import CountVectorizer
        from os.path import join

        did_to_cat = {}
        cat_list = ["CCAT", "GCAT", "MCAT", "ECAT"]
        with open(join(data_dir, "rcv1-v2.topics.qrels")) as fin:
            for line in fin.readlines():
                line = line.strip().split(" ")
                cat = line[0]
                did = int(line[1])
                if cat in cat_list:
                    did_to_cat[did] = did_to_cat.get(did, []) + [cat]
            # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
            for did in list(did_to_cat.keys()):
                if len(did_to_cat[did]) > 1:
                    del did_to_cat[did]

        dat_list = ["lyrl2004_tokens_test_pt0.dat",
                    "lyrl2004_tokens_test_pt1.dat",
                    "lyrl2004_tokens_test_pt2.dat",
                    "lyrl2004_tokens_test_pt3.dat",
                    "lyrl2004_tokens_train.dat"]
        data = []
        target = []
        cat_to_cid = {"CCAT": 0, "GCAT": 1, "MCAT": 2, "ECAT": 3}
        del did
        doc = 0
        did = 0
        for dat in dat_list:
            with open(join(data_dir, dat)) as fin:
                for line in fin.readlines():
                    if line.startswith(".I"):
                        if "did" in locals():
                            assert doc != ""
                            if did in did_to_cat:
                                data.append(doc)
                                target.append(cat_to_cid[did_to_cat[did][0]])
                        did = int(line.strip().split(" ")[1])
                        doc = ""
                    elif line.startswith(".W"):
                        assert doc == ""
                    else:
                        doc += line

        print((len(data), "and", len(did_to_cat)))
        assert len(data) == len(did_to_cat)

        x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
        y = np.asarray(target)

        from sklearn.feature_extraction.text import TfidfTransformer

        x = TfidfTransformer(norm="l2", sublinear_tf=True).fit_transform(x)
        N = how_many or x.shape[0]
        x = x[:N].astype(np.float32)
        print(x.dtype, x.size)
        y = y[:N]
        x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
        print("todense succeed")

        p = np.random.permutation(x.shape[0])
        x = x[p]
        y = y[p]
        print("permutation finished")

        assert x.shape[0] == y.shape[0]
        x = x.reshape((x.shape[0], -1))
        if how_many is None:
            name = 'reuters'
        elif how_many == 10000:
            name = 'reuters10k'
        else:
            name = f'reuters_{how_many}'
        import pickle
        outfile = open(join(data_dir, f"{name}.npy"), 'wb')
        pickle.dump({"data": x, "label": y}, outfile, protocol=4)
        outfile.close()

class GMM_dataset(MyDataset):
    "Synthetic data for visualizations."

    def __init__(self, args, samples=None, labels=None):
        super().__init__(args)
        self.transformer = transforms.Compose([transforms.ToTensor()])
        self._input_dim = 2
        self.k = 15

        if samples:
            self.samples = samples
            self.labels = labels
        else:
            self.mus = [
                    torch.tensor([-5., -5.]),
                    torch.tensor([-5., -3.]),
                    torch.tensor([-5., 0.]),
                    torch.tensor([-5., 3.]),
                    torch.tensor([-5., 5.]),

                    torch.tensor([0., -5.]),
                    torch.tensor([0., -3.]),
                    torch.tensor([0., 0.]),
                    torch.tensor([0., 3.]),
                    torch.tensor([0., 5.]),

                    torch.tensor([5., -5.]),
                    torch.tensor([5., -3.]),
                    torch.tensor([5., 0.]),
                    torch.tensor([5., 3.]),
                    torch.tensor([5., 5.]),
                ]

            # covs = [(torch.rand(2, 2) - 0.5) for _ in range(self.k)]
            covs = [torch.eye(2)*0.5 + (torch.rand(2, 2)*0.5 - 0.3) for _ in range(self.k)]
            self.covs = [covs[k] @ covs[k].T for k in range(self.k)]
            self.weights = torch.div(torch.ones((self.k)), self.k)
            self.GMMs = [
                    torch.distributions.multivariate_normal.MultivariateNormal(loc=self.mus[i], covariance_matrix=self.covs[i])
                for i in range(self.k)
            ]
            self.samples, self.labels = self._sample(10000)


    def _sample(self, n_samples):
        """Sample n_samples from the GMM

        Args:
            n_samples ([int]): Number of samples.
        """
        components = np.random.choice(self.k, n_samples, p=self.weights.numpy())
        samples = [self.GMMs[comp].rsample() for comp in components]

        tensor_samples = torch.cat(
            samples, out=torch.Tensor(n_samples, self.input_dim)
        ).reshape(n_samples, -1)
        return tensor_samples, torch.tensor(components)

    def get_train_data(self):
        return TensorDataset(self.samples, self.labels)

    def get_test_data(self):
        return self.get_train_data()


def merge_datasets(set_1, set_2):
    """
    Merged two TensorDatasets into one
    """
    merged = torch.utils.data.ConcatDataset([set_1, set_2])
    return merged


def generate_mock_dataset(dim, len=3, dtype=torch.float32):
    """Generates a mock TensorDataset

    Args:
        dim (tuple): shape of the sample
        len (int): number of samples. Defaults to 10.
    """
    # Make sure train and test set are of the same type
    if type(dim) == int:
        data = torch.rand((len, dim))
    else:
        data = torch.rand((len, *dim))
    data = torch.tensor(data.clone().detach(), dtype=dtype)
    return TensorDataset(data, torch.zeros(len))
