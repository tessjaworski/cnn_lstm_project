import torch
from torch import nn
from torch_geometric.nn import GCNConv

class CNN_GNN_Hybrid(nn.Module):
    """
    Hybrid model with two branches:
    - ERA5 CNN+LSTM on full grid producing global summary
    - CORA GNN+LSTM on past zeta values producing per-node embeddings
    Outputs per-node predictions for future zeta.

    Inputs:
      - era5_channels: number of ERA5 channels per grid cell
      - cnn_hidden: hidden channels for CNN
      - cnn_lstm_hidden: hidden size for CNN LSTM
      - gcn_hidden: output channels for GCN
      - zeta_lstm_hidden: hidden size for zeta LSTM
      - pred_steps: number of prediction steps per node
    """
    def __init__(
        self,
        era5_channels,
        cnn_hidden,
        cnn_lstm_hidden,
        gcn_hidden,
        zeta_lstm_hidden,
        pred_steps=1,
    ):
        super().__init__()
        # ERA5 CNN branch
        self.era5_cnn = nn.Sequential(
            nn.Conv2d(era5_channels, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_hidden, cnn_hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # compute flattened size dynamically at runtime
        self.cnn_lstm = nn.LSTM(cnn_hidden * 2 * (57 // 4) * (69 // 4), cnn_lstm_hidden, batch_first=True)

        # CORA GNN branch
        self.gcn = GCNConv(1, gcn_hidden)
        self.zeta_lstm = nn.LSTM(gcn_hidden, zeta_lstm_hidden, batch_first=True)

        # Fusion and prediction
        self.fc = nn.Linear(cnn_lstm_hidden + zeta_lstm_hidden, pred_steps)
        self.pred_steps = pred_steps

    def forward(self, era5_seq, zeta_seq, edge_index):
        # era5_seq: [B, T, C, H, W]
        # zeta_seq: [B, T, N]
        B, T, C, H, W = era5_seq.shape
        # ERA5 branch: CNN+LSTM
        # apply cnn per time step
        x = era5_seq.view(B * T, C, H, W)  # [B*T, C, H, W]
        f = self.era5_cnn(x)               # [B*T, 2*cnn_hidden, H/4, W/4]
        f = f.view(B, T, -1)               # [B, T, F_flat]
        out, _ = self.cnn_lstm(f)          # [B, T, cnn_lstm_hidden]
        era5_summary = out[:, -1, :]       # [B, cnn_lstm_hidden]

        # CORA branch: GNN+LSTM
        # prepare node features: past zeta one-hot per timestep
        # zeta_seq shape [B, T, N]
        # process each batch sample separately
        zeta_feats = []
        for b in range(B):
            # for each time, run GCN on single-feature vector
            gcn_seq = []
            for t in range(T):
                z = zeta_seq[b, t].unsqueeze(-1)  # [N, 1]
                h = self.gcn(z, edge_index)       # [N, gcn_hidden]
                gcn_seq.append(h)
            gcn_seq = torch.stack(gcn_seq, dim=1)         # [N, T, gcn_hidden]
            z_out, _ = self.zeta_lstm(gcn_seq)            # [N, T, zeta_lstm_hidden]
            zeta_feats.append(z_out[:, -1, :])           # [N, zeta_lstm_hidden]
        zeta_summary = torch.stack(zeta_feats, dim=0)     # [B, N, zeta_lstm_hidden]

        # fuse: expand era5_summary to per-node
        era5_feat = era5_summary.unsqueeze(1).expand(-1, zeta_summary.size(1), -1)  # [B, N, cnn_lstm_hidden]
        combined = torch.cat([era5_feat, zeta_summary], dim=2)                     # [B, N, sum_hidden]

        # predict
        y = self.fc(combined)   # [B, N, pred_steps]
        out = y.permute(0, 2, 1) # [B, pred_steps, N]
        return out