import torch
from torch import nn
from torch_geometric.nn import GCNConv

class GCNHybrid(nn.Module):
    """

    Inputs:
      - era5_channels: number of ERA5 features per node per timestep
      - gcn_hidden: hidden dimension of the GCN layer
      - lstm_hidden: hidden dimension of both LSTMs
      - pred_steps: number of future time steps to predict
    """
    def __init__(self, era5_channels, gcn_hidden, lstm_hidden, pred_steps=1):
        super().__init__()
        # GCN branch for spatial features
        self.gcn = GCNConv(era5_channels, gcn_hidden)
        self.gcn_lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True)
        # LSTM branch for past zeta sequence
        self.zeta_lstm = nn.LSTM(1, lstm_hidden, batch_first=True)
        # final fully-connected layer to predict pred_steps per node
        self.fc = nn.Linear(lstm_hidden * 2, pred_steps)
        self.pred_steps = pred_steps

    def forward(self, era5_seq, zeta_seq, edge_index):
        # era5_seq: [batch, T, num_nodes, era5_channels]
        # zeta_seq: [batch, T, num_nodes]
        B, T, N, F = era5_seq.shape
        outputs = []
        for b in range(B):
            # spatial branch via GCN + LSTM
            gcn_outs = []
            for t in range(T):
                x_t = era5_seq[b, t]                  # [num_nodes, era5_channels]
                h = self.gcn(x_t, edge_index)         # [num_nodes, gcn_hidden]
                gcn_outs.append(h)
            gcn_seq = torch.stack(gcn_outs, dim=1)    # [num_nodes, T, gcn_hidden]
            gcn_lstm_out, _ = self.gcn_lstm(gcn_seq)  # [num_nodes, T, lstm_hidden]
            spatial_feat = gcn_lstm_out[:, -1, :]     # [num_nodes, lstm_hidden]

            #temporal branch via Zeta LSTM
            z_seq = zeta_seq[b].unsqueeze(-1)         # [T, num_nodes, 1]
            # transpose to [num_nodes, T, 1] for node-wise LSTM
            z_seq = z_seq.permute(1, 0, 2)
            z_lstm_out, _ = self.zeta_lstm(z_seq)     # [num_nodes, T, lstm_hidden]
            zeta_feat = z_lstm_out[:, -1, :]          # [num_nodes, lstm_hidden]

            # combine and predict
            combined = torch.cat([spatial_feat, zeta_feat], dim=1)  # [num_nodes, 2*lstm_hidden]
            y = self.fc(combined)                                  # [num_nodes, pred_steps]
            outputs.append(y.unsqueeze(0))                         # [1, num_nodes, pred_steps]

        out = torch.cat(outputs, dim=0)     # [batch, num_nodes, pred_steps]
        out = out.permute(0, 2, 1)          # [batch, pred_steps, num_nodes]
        return out
