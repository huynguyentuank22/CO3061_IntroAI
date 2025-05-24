import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from model import UltimateTicTacToeModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def preprocess_full_record(row):
    # Channel 1: board state
    board = np.array(list(map(int, row['board_state'].split(',')))).reshape(9, 9)
    
    # Channel 2: macroboard (3x3 → broadcast)
    macro = np.array(list(map(int, row['macroboard'].split(',')))).reshape(3, 3)
    macro_full = np.repeat(np.repeat(macro, 3, axis=0), 3, axis=1)

    # Channel 3: current player
    player = np.full((9, 9), row['current_player'])

    # Channel 4: valid_moves
    valid = np.zeros((9, 9))
    valid_ids = list(map(int, row['valid_moves'].split(',')))
    for move_id in valid_ids:
        i, j = divmod(move_id, 9)
        valid[i][j] = 1

    # Stack all into (4, 9, 9)
    board_tensor = np.stack([board, macro_full, player, valid], axis=0).astype(np.float32)

    # MLP features
    move_number = row['move_number'] / 81.0  # Normalize
    agent_level = row['agent_level']
    game_result = row['game_result']

    mlp_features = np.array([move_number, agent_level, game_result], dtype=np.float32)

    # Label
    label = int(row['chosen_move'])
    
    # mask for valid moves
    mask = np.full(81, -np.inf, dtype=np.float32)
    mask[valid_ids] = 0.0

    # Final result (0 or 1)
    final_result = row['final_result']

    return torch.tensor(board_tensor), torch.tensor(mlp_features), torch.tensor(label), torch.tensor(mask), torch.tensor(final_result)

def prepare_dataset(df):
    board_tensors = []
    mlp_features = []
    labels = []
    masks = []
    final_results = []
    for _, row in df.iterrows():
        board_tensor, mlp_feature, label, mask, final = preprocess_full_record(row)
        board_tensors.append(board_tensor)
        mlp_features.append(mlp_feature)
        labels.append(label)
        masks.append(mask)
        final_results.append(final)

    return torch.stack(board_tensors), torch.stack(mlp_features), torch.tensor(labels), torch.stack(masks), torch.tensor(final_results)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    move_criterion = nn.CrossEntropyLoss()
    result_criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_moves = 0
        correct_results = 0
        total_samples = 0
        
        for board_tensor, mlp_features, labels, masks, final_results in train_loader:
            board_tensor = board_tensor.to(device)
            mlp_features = mlp_features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            final_results = final_results.to(device)
            
            optimizer.zero_grad()
            
            move_logits, result_logits = model(board_tensor, mlp_features)
            
            # Apply mask to move logits (for valid moves only)
            masked_logits = move_logits + masks
            
            move_loss = move_criterion(masked_logits, labels)
            result_loss = result_criterion(result_logits, final_results)
            
            # Combined loss (you can adjust the weighting)
            loss = move_loss + 0.5 * result_loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * board_tensor.size(0)
            
            # Calculate accuracy
            _, move_preds = torch.max(masked_logits, 1)
            _, result_preds = torch.max(result_logits, 1)
            correct_moves += (move_preds == labels).sum().item()
            correct_results += (result_preds == final_results).sum().item()
            total_samples += board_tensor.size(0)
        
        avg_train_loss = total_train_loss / total_samples
        move_accuracy = correct_moves / total_samples
        result_accuracy = correct_results / total_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct_moves = 0
        val_correct_results = 0
        val_total = 0
        
        with torch.no_grad():
            for board_tensor, mlp_features, labels, masks, final_results in val_loader:
                board_tensor = board_tensor.to(device)
                mlp_features = mlp_features.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                final_results = final_results.to(device)
                
                move_logits, result_logits = model(board_tensor, mlp_features)
                
                # Apply mask to move logits
                masked_logits = move_logits + masks
                
                move_loss = move_criterion(masked_logits, labels)
                result_loss = result_criterion(result_logits, final_results)
                
                # Combined loss
                loss = move_loss + 0.5 * result_loss
                
                total_val_loss += loss.item() * board_tensor.size(0)
                
                # Calculate accuracy
                _, move_preds = torch.max(masked_logits, 1)
                _, result_preds = torch.max(result_logits, 1)
                val_correct_moves += (move_preds == labels).sum().item()
                val_correct_results += (result_preds == final_results).sum().item()
                val_total += board_tensor.size(0)
        
        avg_val_loss = total_val_loss / val_total
        val_move_accuracy = val_correct_moves / val_total
        val_result_accuracy = val_correct_results / val_total
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Move Acc: {move_accuracy:.4f} | "
              f"Result Acc: {result_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Move Acc: {val_move_accuracy:.4f} | "
              f"Val Result Acc: {val_result_accuracy:.4f}")
    
    return train_losses, val_losses

def main():
    # Load your data
    df = pd.read_csv('your_data.csv')  # Replace with your data path
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare datasets
    train_board, train_mlp, train_labels, train_masks, train_results = prepare_dataset(train_df)
    val_board, val_mlp, val_labels, val_masks, val_results = prepare_dataset(val_df)
    
    # Create data loaders
    train_dataset = TensorDataset(train_board, train_mlp, train_labels, train_masks, train_results)
    val_dataset = TensorDataset(val_board, val_mlp, val_labels, val_masks, val_results)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = UltimateTicTacToeModel()
    
    # Train model
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=20, 
        learning_rate=0.001
    )
    
    # Save the model
    torch.save(model.state_dict(), 'uttt_model.pth')
    
    # Plot training/validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == "__main__":
    main()
