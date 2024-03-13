from torch.nn import MSELoss
import numpy as np
import torch
import torch.optim as optim
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from torch import from_numpy, tensor
from torch.utils.data import Dataset, DataLoader

pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)


class Dt(Dataset):
    def __init__(self, X, Y):
        self.size = X.shape[0]
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# noinspection PyCallingNonCallable
def multi_encoder(X, Y, d=2, loss_function: object = "MSELoss", neurons=30, epochs=100, batch_size=20, min_global_k=10, max_global_k=1000, seed=0, gpu=False):
    """
    >>> from sklearn import datasets
    >>> from sklearn.preprocessing import StandardScaler
    >>> from numpy import random, round
    >>> digits = datasets.load_digits()
    >>> X = digits.images.reshape((len(digits.images), -1))[:20]
    >>> rnd = random.default_rng(0)
    >>> rnd.shuffle(X)
    >>> X = StandardScaler().fit_transform(X)
    >>> X_ = multi_encoder(X, Y, epochs=2)
    >>> X_.shape
    (20, 2)


    Parameters
    ----------
    X
        Matrix with an instance per row in a given space (often high-dimensional data).
    symmetric
        True:   Take the mean between extrusion and intrusion emphasis.
                See sortedness() documentation for details.
        False:  Weight by original distances (extrusion emphasis), not the projected distances.
    d
        Target dimensionality.
    loss_function
        callable:   Function to minimize.
        "MSELoss":  MSE. Same as `loss_function=MSELoss()` from torch.
    gamma
        Cauchy distribution parameter. Higher values increase the number of neighbors with relevant weight values.
    k
        Number of nearest neighbors to consider for local optimization. This avoids useless sorting of neighbors with insignificant weights (as explained above for parameter `gamma`).
    global_k
        int:    Number of "neighbors" to sample for global optimization.
        "sqrt": Take the square root of the number of points limited by `max_global_k`.
    alpha
        Parameter to balance between local and global. 0 is totally local. 1 is totally global.
    smooothness_tau
        Regularizer. Surrogate function tends to (non differentiable) Kendall tau when smooothness_tau tends to 0.
    neurons
    epochs
    batch_size
    min_global_k
        Lower bound for the number of "neighbors" to sample when `global_k` is dynamic.
    max_global_k
        Upper bound for the number of "neighbors" to sample when `global_k` is dynamic.
    seed
    gpu
        Whether to use GPU.

    Returns
    -------
    Transformed `d`-dimensional data as a numpy float array.

    """

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(X.shape[1], neurons), torch.nn.ReLU(), torch.nn.Linear(neurons, d))
            self.decoder = torch.nn.Sequential(torch.nn.Linear(d, neurons), torch.nn.ReLU(), torch.nn.Linear(neurons, Y.shape[1]))

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    if loss_function == "MSELoss":
        loss_function = MSELoss()
    torch.manual_seed(seed)
    model = M()
    if gpu:
        model.cuda()

    X = X.astype(np.float32)
    X_ = from_numpy(X).cuda() if gpu else from_numpy(X)
    Y_ = from_numpy(Y).cuda() if gpu else from_numpy(Y)

    optimizer = optim.RMSprop(model.parameters())
    model.train()
    loader = DataLoader(Dt(X_, Y_), shuffle=True, batch_size=batch_size, pin_memory=gpu)
    with torch.enable_grad():
        for i in range(epochs):
            for Xb, Yb in loader:
                Xb_, Zb = model(Xb)
                loss = loss_function(Xb_, Zb)
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()
    return model(X)[0].detach().cpu().numpy().astype(float)
