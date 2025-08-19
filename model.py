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
        pred_steps=1
    ):
        super().__init__()
        # dropout layers
        self.dropout_lstm_cnn = nn.Dropout(p=0.5)
        self.dropout_lstm_zeta = nn.Dropout(p=0.5)
        self.dropout_fc = nn.Dropout(p=0.5)
        
        # ERA5 CNN branch
        self.era5_cnn = nn.Sequential(
            nn.Conv2d(era5_channels, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
            nn.Conv2d(cnn_hidden, cnn_hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )
        # flattened size assumes ERA5 inputs are 57x69 (Gulf of Mexico bounding box)
        # becomes (57//4 x 69//4) after two pooling layers
        self.cnn_lstm = nn.LSTM(cnn_hidden * 2 * (57 // 4) * (69 // 4), cnn_lstm_hidden, batch_first=True)

        # CORA GNN branch
        self.gcn = GCNConv(1, gcn_hidden)
        self.zeta_lstm = nn.LSTM(gcn_hidden, zeta_lstm_hidden, batch_first=True)

        # Fusion and prediction
        self.fc = nn.Linear(cnn_lstm_hidden + zeta_lstm_hidden, pred_steps)
        self.pred_steps = pred_steps

    def forward(self, era5_seq, zeta_seq, edge_index):
        # B: batch size (num of sequences)
        # T: time steps in the sequence
        # C: ERA5 features per grid point
        # H,W: spatial height and width
        # era5_seq: [B, T, C, H, W]
        # zeta_seq: [B, T, N]
        B, T, C, H, W = era5_seq.shape # unpacks shape of ERA5 tensor

        # apply cnn independetly to each time step
        x = era5_seq.view(B * T, C, H, W)  # reshape to merge batch and time 
        f = self.era5_cnn(x)               # applies cnn to extract spatial features
        f = f.view(B, T, -1)               # reshapes cnn output back into a sequence
        out, _ = self.cnn_lstm(f)          # passes sequence of cnn features into an lstm
        out = self.dropout_lstm_cnn(out)
        era5_summary = out[:, -1, :]       # takes full temporal context of ERA5 input

        # CORA branch: GNN+LSTM
        # prepare node features: past zeta values per timestep
        # zeta_seq shape [B, T, N]
        # process each batch sample separately
        zeta_feats = []
        for b in range(B):
            # for each time, run GCN on single-feature vector
            gcn_seq = []
            for t in range(T): # for each time step extract the data vector
                z = zeta_seq[b, t].unsqueeze(-1)
                h = self.gcn(z, edge_index)       # pass zeta snapshot through gcn that uses edge_index to gather info from neighbors
                gcn_seq.append(h) # stack gcn outputs across time
            gcn_seq = torch.stack(gcn_seq, dim=1)
            z_out, _ = self.zeta_lstm(gcn_seq) # feed into lstm for temporal dependencies
            z_out = self.dropout_lstm_zeta(z_out)
            zeta_feats.append(z_out[:, -1, :])           # keep final temporal encoding
        zeta_summary = torch.stack(zeta_feats, dim=0)     # stack features from all samples

        # fuse: broadcast global ERA5 summary to each node and concatenate with node-specific zeta features
        era5_feat = era5_summary.unsqueeze(1).expand(-1, zeta_summary.size(1), -1)  # one summary vector per node per sample
        combined = torch.cat([era5_feat, zeta_summary], dim=2) # concat global context and node-specific zeta temporal features
        combined = self.dropout_fc(combined)

        # predict
        y = self.fc(combined)   # maps feature vector to a prediction vector
        out = y.permute(0, 2, 1) # transpose to match expected shape [B, pred_steps, num_nodes]
        return out # model's output: for each time step in the future, it outputs the predicted zeta values at all nodes
