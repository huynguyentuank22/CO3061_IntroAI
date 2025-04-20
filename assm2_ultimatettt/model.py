import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class UltimateTicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN part
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)  # Input: (4,9,9)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Flatten output from CNN
        self.flatten_cnn = nn.Linear(64 * 9 * 9, 256)
        # Batch normalization
        self.bn_cnn = nn.BatchNorm1d(256)
        # Dropout
        self.dropout_cnn = nn.Dropout(0.3)
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Fusion + output
        self.fc_final = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 81)  # 81 possible moves
        )

    def forward(self, board_tensor, mlp_features):
        x = F.relu(self.conv1(board_tensor))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.flatten_cnn(x)

        y = self.mlp(mlp_features)

        out = self.fc_final(torch.cat([x, y], dim=1))
        return out
