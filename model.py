import torch
from torch import nn
from torch_geometric.nn import GCNConv
from cora_graph import build_edge_index, load_cora_coordinates

class HybridCNNLSTM(nn.Module):
    #era5_channels is total input features
    #out_channels is just the zeta prediction
    def __init__(self, era5_channels, zeta_nodes, coords, k_neighbors=8, pred_steps=24, height=57, width=69, lstm_hidden_era=64, lstm_hidden_node=64):
        super(HybridCNNLSTM, self).__init__()
        edge_index = build_edge_index(coords, k=k_neighbors)
        self.register_buffer("edge_index", edge_index)

        #cnn block
        #handles each individual map at each timestep
        # at each time step, conv2D extracts local spatial features
        self.era5_cnn = nn.Sequential(
            nn.Conv2d(era5_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(), # adds non-linearity
            nn.MaxPool2d(2),  # downsamples the image
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # compute CNN output size: 64 × (H/4) × (W/4)
        cnn_h = height // 4
        cnn_w = width  // 4
        cnn_out_size = 64 * cnn_h * cnn_w
        
        #lstm block
        #processes sequence of cnn outputs across timesteps
        # learns how spatial patterns change over time
        self.era5_lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=lstm_hidden_era,
            batch_first=True
        )

        self.era5_out_dim = lstm_hidden_era

        # zeta block
        # GNN
        self.zeta_nodes = zeta_nodes

        self.gnn1 = GCNConv(in_channels=1, out_channels=32)
        self.gnn2 = GCNConv(in_channels=32, out_channels=64)

        #zeta lstm
        self.zeta_lstm = nn.LSTM(
            input_size=64, #gnn 64 dim node vectors
            hidden_size=lstm_hidden_node,
            batch_first=True
        )
        self.node_out_dim = lstm_hidden_node
        self.pred_steps = pred_steps
        merged_dim = self.era5_out_dim + self.node_out_dim
        # Merge both processed inputs
        self.combined_fc = nn.Sequential(
            nn.Linear(merged_dim, 512),
            nn.ReLU(),
            nn.Linear(512, zeta_nodes * pred_steps)
        )

    def forward(self, era5_seq, zeta_seq):
        # era5_seq: (batch, time, channels, height, width)
        # unpack the input shape
        B, T, C, H, W = era5_seq.shape
        # merge batch and time for cnn so it can be applied to each timestep independently
        era5_seq = era5_seq.view(B * T, C, H, W)
        era5_features = self.era5_cnn(era5_seq)  # run cnn per time step
        # flatten spatial features for LSTM
        era5_features = era5_features.view(B, T, -1)  # (B, T, flattened_features)

        #run lstm
        lstm_out, _ = self.era5_lstm(era5_features) #lstm_out is sequence of hidden states
        #grab the last timestep's output (learned summary of the whole sequence)
        era5_summary = lstm_out[:, -1, :]  # (B, lstm_hidden)


       # Zeta branch
       # gnn
        all_node_embeddings = []
        for t in range(T):
            z_t = zeta_seq[:, t, :]  
            flat_feats = z_t.reshape(B * zeta_seq.size(-1), 1)
            edge_index_batch = self.edge_index.repeat(1, B)
            batch_offsets = (
                torch.arange(B, device=z_t.device).unsqueeze(1)
                * zeta_seq.size(-1)
            )
            edge_index_batch = edge_index_batch + batch_offsets.repeat(1, edge_index_batch.size(1) // B).flatten()
            h = self.gnn1(flat_feats, edge_index_batch)
            h = torch.relu(h)
            h = self.gnn2(h, edge_index_batch)
            h = torch.relu(h)
            h = h.view(B, zeta_seq.size(-1), -1)
            node_summary = h.mean(dim=1) 
            all_node_embeddings.append(node_summary)

        #lstm
        zeta_over_time = torch.stack(all_node_embeddings, dim=1)
        z_lstm_out, _ = self.zeta_lstm(zeta_over_time) 
        zeta_summary = z_lstm_out[:, -1, :]  


        # Combine and predict next zeta
        combined = torch.cat([era5_summary, zeta_summary], dim=1)
        out = self.combined_fc(combined)  # (B, zeta_nodes)
        out = out.view(B, self.pred_steps, -1) #reshape
        return out
