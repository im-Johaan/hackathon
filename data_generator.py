import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")
    parser.add_argument("--num_entities", type=int, default=50, help="Number of entities")
    parser.add_argument("--num_events", type=int, default=10000, help="Number of total events (approx)")
    parser.add_argument("--anomaly_rate", type=float, default=0.03, help="Percentage of anomalies")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    return parser.parse_args()

def random_ip(fake):
    return fake.ipv4()

def random_device_fingerprint(fake, os_type=None):
    os_types = ["Windows", "macOS", "Linux", "iOS", "Android", "EmbeddedOS"]
    if not os_type:
        os_type = random.choice(os_types)
    mac = fake.mac_address()
    firmware = f"v{random.randint(1,5)}.{random.randint(0,9)}"
    return f"{os_type}|{mac}|{firmware}"

def generate_data(args):
    fake = Faker()
    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    
    ENTITY_TYPES = ["User", "Service Account", "Edge Device"]
    RESOURCES = ["VPN", "Database", "FileServer", "WebPortal", "API", "SSH", "GitLab"]
    AUTH_METHODS = ["Password", "MFA", "Token", "Cert", "Biometric"]
    GEO_LOCATIONS = ["US-East", "US-West", "EU-Central", "Asia-South", "Asia-East"]
    
    # 1. Profiles
    profiles = {}
    for i in range(args.num_entities):
        e_type = np.random.choice(ENTITY_TYPES, p=[0.7, 0.2, 0.1])
        e_id = f"{e_type[:3].upper()}_{fake.uuid4()[:8]}"
        profiles[e_id] = {
            "entity_type": e_type,
            "primary_geo": random.choice(GEO_LOCATIONS),
            "typical_resources": random.sample(RESOURCES, k=random.randint(1, 3)),
            "typical_auth": random.choice(AUTH_METHODS),
            "device_fingerprint": random_device_fingerprint(fake, os_type="Linux" if e_type == "Edge Device" else None),
            "primary_ip": random_ip(fake)
        }
        
    entity_ids = list(profiles.keys())
    events = []
    
    current_time = datetime(2023, 1, 1, 8, 0, 0)
    
    anomaly_types = [
        "brute_force",
        "impossible_travel",
        "credential_stuffing",
        "lateral_movement",
        "device_spoofing",
        "low_and_slow_exfiltration",
        "insider_drift"
    ]
    
    target_anomalies = int(args.num_events * args.anomaly_rate)
    target_normal = args.num_events - target_anomalies
    
    # Generate Normal Baseline
    for _ in range(target_normal):
        e_id = random.choice(entity_ids)
        p = profiles[e_id]
        current_time += timedelta(minutes=random.randint(1, 60))
        
        event = {
            "entity_id": e_id,
            "entity_type": p["entity_type"],
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": p["primary_ip"] if random.random() > 0.1 else random_ip(fake),
            "geo_location": p["primary_geo"],
            "resource_accessed": random.choice(p["typical_resources"]),
            "auth_method": p["typical_auth"],
            "session_duration": max(1, int(np.random.normal(300, 100))),
            "command_sequence": "NORMAL_OP",
            "device_fingerprint": p["device_fingerprint"],
            "label": "normal"
        }
        events.append(event)

    # Generate Anomalies
    a_counts = {t: max(1, target_anomalies // len(anomaly_types)) for t in anomaly_types}
    
    for a_type, count in a_counts.items():
        for _ in range(count):
            e_id = random.choice(entity_ids)
            p = profiles[e_id]
            current_time += timedelta(minutes=random.randint(1, 60))
            
            base_event = {
                "entity_id": e_id,
                "entity_type": p["entity_type"],
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_ip": p["primary_ip"],
                "geo_location": p["primary_geo"],
                "resource_accessed": random.choice(p["typical_resources"]),
                "auth_method": p["typical_auth"],
                "session_duration": 300,
                "command_sequence": "NORMAL_OP",
                "device_fingerprint": p["device_fingerprint"],
                "label": a_type
            }
            
            if a_type == "brute_force":
                for i in range(5):
                    ev = base_event.copy()
                    ev["timestamp"] = (current_time + timedelta(seconds=i*2)).strftime("%Y-%m-%d %H:%M:%S")
                    ev["session_duration"] = 0
                    ev["command_sequence"] = "FAILED_AUTH"
                    events.append(ev)
            elif a_type == "impossible_travel":
                ev1 = base_event.copy()
                events.append(ev1)
                ev2 = base_event.copy()
                ev2["timestamp"] = (current_time + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
                other_geos = list(set(GEO_LOCATIONS) - {p["primary_geo"]})
                ev2["geo_location"] = random.choice(other_geos) if other_geos else "Antarctica"
                ev2["source_ip"] = random_ip(fake)
                events.append(ev2)
            elif a_type == "credential_stuffing":
                attacker_ip = random_ip(fake)
                for i in range(5):
                    t_id = random.choice(entity_ids)
                    ev = base_event.copy()
                    ev["entity_id"] = t_id
                    ev["entity_type"] = profiles[t_id]["entity_type"]
                    ev["timestamp"] = (current_time + timedelta(seconds=i*2)).strftime("%Y-%m-%d %H:%M:%S")
                    ev["source_ip"] = attacker_ip
                    ev["session_duration"] = 0
                    ev["command_sequence"] = "FAILED_AUTH"
                    events.append(ev)
            elif a_type == "lateral_movement":
                other_res = list(set(RESOURCES) - set(p["typical_resources"]))
                ev = base_event.copy()
                ev["resource_accessed"] = random.choice(other_res) if other_res else "SecretVault"
                ev["command_sequence"] = "SCAN_NETWORK"
                events.append(ev)
            elif a_type == "device_spoofing":
                ev = base_event.copy()
                ev["device_fingerprint"] = random_device_fingerprint(fake)
                events.append(ev)
            elif a_type == "low_and_slow_exfiltration":
                ev = base_event.copy()
                ev["timestamp"] = current_time.replace(hour=random.randint(0, 4)).strftime("%Y-%m-%d %H:%M:%S") # Off-hours
                ev["session_duration"] = random.randint(1000, 5000)
                ev["command_sequence"] = "DB_DUMP -> ZIP -> FTP_OUT"
                events.append(ev)
            elif a_type == "insider_drift":
                other_res = list(set(RESOURCES) - set(p["typical_resources"]))
                ev = base_event.copy()
                ev["resource_accessed"] = random.choice(other_res) if other_res else "AdminPanel"
                events.append(ev)

    df = pd.DataFrame(events)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Save datasets
    full_path = os.path.join(args.out_dir, "full_labeled_dataset.csv")
    df.to_csv(full_path, index=False)
    
    inf_df = df.drop(columns=["label"])
    inf_path = os.path.join(args.out_dir, "inference_dataset.csv")
    inf_df.to_csv(inf_path, index=False)
    
    # Sanity checks
    print("--- DATA GENERATION SANITY CHECKS ---")
    print(f"Total events: {len(df)}")
    print("\nCounts per label:")
    print(df["label"].value_counts().to_string())
    print("\nCounts per entity_type:")
    print(df["entity_type"].value_counts().to_string())
    
    print("\nExample rows per attack type:")
    for label in df["label"].unique():
        print(f"\n[{label.upper()}]")
        print(df[df["label"] == label].head(2).to_string(index=False))

if __name__ == "__main__":
    args = parse_args()
    generate_data(args)
