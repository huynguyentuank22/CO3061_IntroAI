import torch.nn as nn
import torch.nn.functional as F
import torch

class UltimateTicTacToeModel(nn.Module):
    def __init__(self):
        super(UltimateTicTacToeModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(32 * 81 + 32, 256),
            nn.ReLU()
        )
        
        self.output_moves = nn.Linear(256, 81)        # dự đoán nước đi
        self.output_result = nn.Linear(256, 2)        # dự đoán final_result (0: thua/hòa, 1: thắng)

    def forward(self, board_tensor, mlp_features):
        x1 = self.cnn(board_tensor)                   # (B, 32*81)
        x2 = self.fc_mlp(mlp_features)                # (B, 32)
        x = torch.cat([x1, x2], dim=1)                # (B, 32*81 + 32)
        x = self.combined(x)

        move_logits = self.output_moves(x)            # (B, 81)
        result_logits = self.output_result(x)         # (B, 2)
        return move_logits, result_logits