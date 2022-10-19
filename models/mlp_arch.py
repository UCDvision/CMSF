import torch.nn as nn


def get_mlp(inp_dim, hidden_dim, out_dim):
    """Returns a two-layer MLP; used for projection layers and classifier.

    """
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp


def get_mlp_3l(inp_dim, hidden_dim_1, hidden_dim_2, out_dim):
    """Returns a two-layer MLP; used for classifier.

    """
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim_1),
        nn.BatchNorm1d(hidden_dim_1),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim_1, hidden_dim_2),
        nn.BatchNorm1d(hidden_dim_2),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim_2, out_dim),
    )
    return mlp


