import torch
from torch import nn

class HybridCNNLSTM(nn.Module):
    #in_channels is total input features (ERA5 + CORA)
    #out_channels is just the zeta prediction
    def __init__(self, era5_channels, zeta_nodes, height=57, width=69, lstm_hidden=128):
        super(HybridCNNLSTM, self).__init__()

        #cnn block
        #handles each individual map at each timestep
        # at each time step, conv2D extracts local spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(era5_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(), # adds non-linearity
            nn.MaxPool2d(2),  # downsamples the image
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, era5_channels, height, width)
            out = self.era5_cnn(dummy)
            self.cnn_out_size = out.view(1, -1).shape[1]
        
        #lstm block
        #processes sequence of cnn outputs across timesteps
        # learns how spatial patterns change over time
        self.era5_lstm = nn.LSTM(
            input_size=self.cnn_out_size,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # compress CORA zeta sequence with FC
        self.zeta_fc = nn.Sequential(
            nn.Linear(zeta_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Merge both processed inputs
        self.combined_fc = nn.Sequential(
            nn.Linear(lstm_hidden + 256, 512),
            nn.ReLU(),
            nn.Linear(512, zeta_nodes)  # predict full zeta map
        )

        def forward(self, era5_seq, zeta_seq):
            # era5_seq: (batch, time, channels, height, width)
            # unpack the input shape
            B, T, C, H, W = x.shape
            # merge batch and time for cnn so it can be applied to each timestep independently
            era5_seq = era5_seq.view(B * T, C, H, W)
            era5_features = self.era5_cnn(era5_seq)  # run cnn per time step
            # flatten spatial features for LSTM
            era5_features = era5_features.view(B, T, -1)  # (B, T, flattened_features)

            #run lstm
            lstm_out, _ = self.era5_lstm(era5_features) #lstm_out is sequence of hidden states
            #grab the last timestep's output (learned summary of the whole sequence)
            era5_summary = lstm_out[:, -1, :]  # (B, lstm_hidden)

            # Process zeta sequence (B, T, nodes) → just last timestep for now
            zeta_last = zeta_seq[:, -1, :]  # (B, nodes)
            zeta_embed = self.zeta_fc(zeta_last)  # (B, 256)

            # Combine and predict next zeta
            combined = torch.cat([era5_summary, zeta_embed], dim=1)
            out = self.combined_fc(combined)  # (B, zeta_nodes)
            return out
