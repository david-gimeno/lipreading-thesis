import torch
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

class Adapter(torch.nn.Module):
    """Adapter module definition based on Bapna and Firat (2019).

    :param int idim: input dimension
    :param int hidden_units: number of hidden units
    """

    def __init__(self, idim, hidden_units):
        super(Adapter, self).__init__()

        self.ln = LayerNorm(idim)
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, x):
        """Forward pass.
        """
        return self.w_2(torch.relu(self.w_1(self.ln(x)))) + x
