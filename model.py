import torch
from torch import nn

class HybridCNNLSTM(nn.Module):
    #era5_channels is total input features
    #out_channels is just the zeta prediction
    def __init__(self, era5_channels, zeta_nodes, pred_steps=3, height=57, width=69, lstm_hidden=128):
        super(HybridCNNLSTM, self).__init__()

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
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # lstm block
        self.zeta_lstm = nn.LSTM(
            input_size=zeta_nodes,
            hidden_size=64,
            batch_first=True
        )
        self.pred_steps = pred_steps
        # Merge both processed inputs
        self.combined_fc = nn.Sequential(
            nn.Linear(lstm_hidden + 64, 512),
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
        z, _ = self.zeta_lstm(zeta_seq)
        zeta_feat = z[:, -1, :]    


        # Combine and predict next zeta
        combined = torch.cat([era5_summary, zeta_feat], dim=1)
        out = self.combined_fc(combined)  # (B, zeta_nodes)
        out = out.view(B, self.pred_steps, -1) #reshape
        return out