import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading session scores...")
    df = pd.read_csv("session_scores.csv")
    
    # Define features
    feature_cols = [
        'hour_of_day', 'day_of_week', 'is_cold_start', 
        'time_since_last_session_hrs', 'distance_from_last_km', 
        'implied_travel_speed_kmh', 'failed_auth_entity_15m', 
        'failed_auth_ip_15m_distinct', 'is_new_resource', 
        'new_resources_last_10', 'is_off_hours', 'device_differs', 
        'os_differs', 'mac_differs', 'firmware_differs', 'session_duration_z'
    ]
    
    # 1. Train Decision Tree on all non-normal data to learn attack signatures
    print("Training Decision Tree on ground-truth attacks for feature importance...")
    df_attacks = df[df['label'] != 'normal'].copy()
    
    X_train = df_attacks[feature_cols].fillna(0)
    y_train = df_attacks['label']
    
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    
    # Report feature importances
    importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nDecision Tree Feature Importances for Attack Classification:")
    print(importances[importances > 0.01].to_string())
    
    # 2. Select Top 1% Alerts based on anomaly_score
    num_alerts = max(1, int(0.01 * len(df)))
    threshold = df['anomaly_score'].nlargest(num_alerts).iloc[-1]
    
    df_alerts = df[df['anomaly_score'] >= threshold].copy()
    print(f"\nProcessing {len(df_alerts)} alerts (Top 1% Budget, Threshold >= {threshold:.4f})...")
    
    # 3. Classify and Explain
    X_alerts = df_alerts[feature_cols].fillna(0)
    preds = dt.predict(X_alerts)
    probs = np.max(dt.predict_proba(X_alerts), axis=1)
    
    # Apply threshold for unclassified anomaly
    preds = np.where(probs < 0.6, 'unclassified_anomaly', preds)
    
    df_alerts['predicted_attack_type'] = preds
    df_alerts['confidence'] = probs
    
    # Generate explanations
    explanations = []
    for idx, row in df_alerts.iterrows():
        pred = row['predicted_attack_type']
        exp = f"Flagged by sequence model (Score: {row['anomaly_score']:.2f}). "
        
        if pred == 'brute_force':
            exp += f"Matches brute force: {int(row['failed_auth_entity_15m'])} failed auths for this entity in 15m."
        elif pred == 'credential_stuffing':
            exp += f"Matches credential stuffing: {int(row['failed_auth_ip_15m_distinct'])} distinct entities failed auth from IP {row['source_ip']} in 15m."
        elif pred == 'impossible_travel':
            exp += f"Matches impossible travel: implied speed {row['implied_travel_speed_kmh']:.0f} km/h over {row['distance_from_last_km']:.0f} km."
        elif pred == 'device_spoofing':
            exp += f"Matches device spoofing: Fingerprint mismatch."
        elif pred == 'low_and_slow_exfiltration':
            exp += f"Matches low-and-slow exfiltration: Off-hours access with duration z-score {row['session_duration_z']:.1f}."
        elif pred == 'lateral_movement':
            exp += f"Matches lateral movement: Accessed new resource '{row['resource_accessed']}', {int(row['new_resources_last_10'])} new resources recently."
        elif pred == 'insider_drift':
            exp += f"Matches insider drift (ambiguous): Accessed unusual resource '{row['resource_accessed']}'."
        else:
            exp += "Unclassified anomaly pattern."
            
        explanations.append(exp)
        
    df_alerts['explanation_text'] = explanations
    df_alerts['event_id'] = df_alerts.index # using index as event_id
    
    # Format for output
    cols_to_keep = ['event_id', 'entity_id', 'timestamp', 'anomaly_score', 'predicted_attack_type', 'confidence', 'explanation_text'] + feature_cols
    df_scored_alerts = df_alerts[cols_to_keep]
    
    # 4. Evaluate Classification Accuracy against Ground Truth
    y_true_alerts = df_alerts['label'].copy()
    y_pred_alerts = df_alerts['predicted_attack_type'].copy()
    
    print("\n--- Alert Classification Evaluation (Top 1% Alerts) ---")
    
    all_classes = sorted(list(set(y_true_alerts.unique()) | set(y_pred_alerts.unique())))
    
    print("Classification Report:")
    print(classification_report(y_true_alerts, y_pred_alerts, labels=all_classes, zero_division=0))
    
    print("Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    cm = confusion_matrix(y_true_alerts, y_pred_alerts, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    print(cm_df.to_string())
    
    print("\nNote: 'insider_drift' is an ambiguous edge case used for false-positive tuning, NOT a confirmed attack.")
    
    # Save output
    df_scored_alerts.to_csv("scored_alerts.csv", index=False)
    print("\nSaved 1% alerts to scored_alerts.csv")

if __name__ == "__main__":
    main()
