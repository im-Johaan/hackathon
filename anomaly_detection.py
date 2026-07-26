import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse

# Config
WINDOW_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_DIM = 32

class GRUAutoencoder(nn.Module):
    # Using GRU because it typically trains faster and requires fewer parameters than LSTM, 
    # while matching performance on tabular sequential data like access logs.
    def __init__(self, input_dim, hidden_dim):
        super(GRUAutoencoder, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, hidden = self.encoder(x)
        # hidden: (1, batch, hidden_dim)
        
        # repeat hidden state for seq_len
        decoder_input = hidden.permute(1, 0, 2).repeat(1, x.size(1), 1)
        
        decoder_out, _ = self.decoder(decoder_input)
        out = self.output_layer(decoder_out)
        return out

def create_sequences(df, feature_cols, window_size):
    """
    Creates sequences per entity.
    Returns:
    - sequences: (N, window_size, num_features)
    - indices: (N,) corresponding to the index of the *last* event in the window
    """
    sequences = []
    indices = []
    
    # Sort just in case
    df = df.sort_values(['entity_id', 'timestamp'])
    grouped = df.groupby('entity_id')
    
    for entity_id, group in grouped:
        group_idx = group.index.values
        group_features = group[feature_cols].values
        
        if len(group) >= window_size:
            for i in range(len(group) - window_size + 1):
                seq = group_features[i:i+window_size]
                idx = group_idx[i+window_size-1]
                sequences.append(seq)
                indices.append(idx)
    
    if len(sequences) == 0:
        return np.array([]), np.array([])
    return np.stack(sequences), np.array(indices)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_x in train_loader:
        batch_x = batch_x[0].to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader) if len(train_loader) > 0 else 0

def get_reconstruction_error(model, data_loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            output = model(batch_x)
            # MSE per sequence
            mse = torch.mean((output - batch_x)**2, dim=(1,2)).cpu().numpy()
            errors.extend(mse)
    return np.array(errors)

def evaluate_and_plot(df, scores_col, label_col):
    # Map all attacks to 1, normal to 0
    y_true = (df[label_col] != 'normal').astype(int)
    y_scores = df[scores_col].fillna(0)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Calculate top 1% alert budget threshold
    num_alerts = int(0.01 * len(df))
    sorted_scores = np.sort(y_scores)[::-1]
    if num_alerts > 0:
        threshold_1pct = sorted_scores[num_alerts-1]
    else:
        threshold_1pct = sorted_scores[0]
        
    y_pred_1pct = (y_scores >= threshold_1pct).astype(int)
    
    print("\n--- Evaluation Metrics ---")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Top 1% Alert Threshold: {threshold_1pct:.4f}")
    print("\nClassification Report (Top 1% Alert Budget):")
    print(classification_report(y_true, y_pred_1pct, target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Breakdown by Attack Type:")
    # Breakdown of alerted vs non-alerted by ground truth label
    df_eval = pd.DataFrame({'label': df[label_col], 'alerted': y_pred_1pct})
    breakdown = df_eval.groupby('label')['alerted'].agg(['count', 'sum'])
    breakdown['recall'] = breakdown['sum'] / breakdown['count']
    print(breakdown.rename(columns={'count': 'Total', 'sum': 'Detected'}))
    
    # Plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=scores_col, hue=y_true, bins=50, log_scale=(False, True))
    plt.axvline(threshold_1pct, color='r', linestyle='--', label='Top 1% Threshold')
    plt.title("Anomaly Score Distribution")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, marker='.')
    plt.title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.tight_layout()
    plt.savefig("evaluation_report.png")
    print("Saved plots to evaluation_report.png")

def main():
    print("Loading engineered features...")
    df = pd.read_csv("engineered_features_labeled.csv")
    
    # Convert bools to int if needed
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            
    # Define features
    feature_cols = [
        'hour_of_day', 'day_of_week', 'is_cold_start', 
        'time_since_last_session_hrs', 'distance_from_last_km', 
        'implied_travel_speed_kmh', 'failed_auth_entity_15m', 
        'failed_auth_ip_15m_distinct', 'is_new_resource', 
        'new_resources_last_10', 'is_off_hours', 'device_differs', 
        'os_differs', 'mac_differs', 'firmware_differs', 'session_duration_z'
    ]
    
    # Fill NAs
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Split chronologically: First 50% for initial train, next 50% for inference + concept drift updates
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df) * 0.5)
    
    df_init_train = df.iloc[:split_idx].copy()
    df_eval_stream = df.iloc[split_idx:].copy()
    
    # Initial Training Set: ONLY benign windows
    df_init_train_benign = df_init_train[df_init_train['label'] == 'normal'].copy()
    
    scaler = StandardScaler()
    df_init_train_benign.loc[:, feature_cols] = scaler.fit_transform(df_init_train_benign[feature_cols])
    
    print("Building sequences for initial training...")
    X_train, _ = create_sequences(df_init_train_benign, feature_cols, WINDOW_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUAutoencoder(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if len(X_train) > 0:
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
        
        print("Training initial autoencoder on benign data...")
        for epoch in range(EPOCHS):
            loss = train_model(model, train_loader, criterion, optimizer, device)
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")
    
    # Baseline population error (for cold start fallback)
    # Average reconstruction error over the initial training set
    if len(X_train) > 0:
        base_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=False)
        base_errors = get_reconstruction_error(model, base_loader, device)
        population_avg_error = np.mean(base_errors)
    else:
        population_avg_error = 0.5
        
    print(f"Population avg reconstruction error (for cold start): {population_avg_error:.4f}")
    
    # Initialize score column
    df['anomaly_score'] = np.nan
    
    # Apply to entire dataset to get scores
    # We will simulate a streaming/chunked process for the eval portion to handle concept drift.
    
    # First, score the first half (just to have complete scores)
    df_scaled_first = df_init_train.copy()
    df_scaled_first[feature_cols] = scaler.transform(df_scaled_first[feature_cols])
    X_first, idx_first = create_sequences(df_scaled_first, feature_cols, WINDOW_SIZE)
    if len(X_first) > 0:
        loader = DataLoader(TensorDataset(torch.tensor(X_first, dtype=torch.float32)), batch_size=BATCH_SIZE)
        errors = get_reconstruction_error(model, loader, device)
        df.loc[df_init_train.index[idx_first], 'anomaly_score'] = errors
        
    # Streaming eval over the second half in chunks of 1000 events
    chunk_size = 1000
    eval_indices = df_eval_stream.index.values
    
    for start_i in range(0, len(eval_indices), chunk_size):
        end_i = min(start_i + chunk_size, len(eval_indices))
        chunk_idx = eval_indices[start_i:end_i]
        
        # Need context for sequences, so we take df up to end_i
        current_df = df.iloc[:end_i+1].copy() # +1 because end_i is exclusive index
        current_df[feature_cols] = scaler.transform(current_df[feature_cols])
        
        # For efficiency, we can filter current_df to only entities present in chunk.
        entities_in_chunk = df.loc[chunk_idx, 'entity_id'].unique()
        current_df_filtered = current_df[current_df['entity_id'].isin(entities_in_chunk)].copy()
        
        X_chunk, idx_chunk = create_sequences(current_df_filtered, feature_cols, WINDOW_SIZE)
        
        # Find which idx_chunk belong to our chunk_idx
        mask = np.isin(idx_chunk, chunk_idx)
        if mask.any():
            X_chunk_target = X_chunk[mask]
            idx_chunk_target = idx_chunk[mask]
            
            loader = DataLoader(TensorDataset(torch.tensor(X_chunk_target, dtype=torch.float32)), batch_size=BATCH_SIZE)
            errors = get_reconstruction_error(model, loader, device)
            df.loc[idx_chunk_target, 'anomaly_score'] = errors
            
        # CONCEPT DRIFT UPDATE:
        # Fine-tune the model on the purely benign sequences from this chunk
        # to adapt to new normal behaviors
        benign_chunk_idx = df.loc[chunk_idx]
        benign_chunk_idx = benign_chunk_idx[benign_chunk_idx['label'] == 'normal'].index.values
        
        mask_benign = np.isin(idx_chunk, benign_chunk_idx)
        if mask_benign.any():
            X_finetune = X_chunk[mask_benign]
            ft_loader = DataLoader(TensorDataset(torch.tensor(X_finetune, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
            # Fine-tune for 1 epoch
            train_model(model, ft_loader, criterion, optimizer, device)
            
    # Handle cold start and missing scores
    missing_mask = df['anomaly_score'].isna()
    
    # Simple heuristic fallback for cold start: 
    # Base error + penalty if device differs or new resource accessed or distance is far
    penalty = df['device_differs'] * 0.5 + df['is_new_resource'] * 0.5 + (df['implied_travel_speed_kmh'] > 800).astype(float) * 1.0
    df.loc[missing_mask, 'anomaly_score'] = population_avg_error + penalty[missing_mask]
    
    # Ensure no NaNs
    df['anomaly_score'] = df['anomaly_score'].fillna(population_avg_error)
    
    print("\nScoring complete. Saving model and scores...")
    torch.save(model.state_dict(), "anomaly_detector_gru.pt")
    
    # Save the scores
    df.to_csv("session_scores.csv", index=False)
    print("Saved session_scores.csv and anomaly_detector_gru.pt")
    
    # Evaluate
    evaluate_and_plot(df, 'anomaly_score', 'label')

if __name__ == "__main__":
    main()
