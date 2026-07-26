import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
import math

def calculate_distance(geo1, geo2):
    coords = {
        "US-East": (39.0, -77.0),
        "US-West": (37.0, -122.0),
        "EU-Central": (50.0, 8.0),
        "Asia-South": (19.0, 72.0),
        "Asia-East": (35.0, 139.0),
        "Antarctica": (-82.0, 0.0)
    }
    
    if geo1 not in coords or geo2 not in coords:
        return 0.0
    if geo1 == geo2:
        return 0.0
        
    lat1, lon1 = coords[geo1]
    lat2, lon2 = coords[geo2]
    
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r

def process_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    global_stats = {}
    for etype in df['entity_type'].unique():
        mask = df['entity_type'] == etype
        global_stats[etype] = {
            'mean_dur': df.loc[mask, 'session_duration'].mean(),
            'std_dur': df.loc[mask, 'session_duration'].std() + 1e-5,
            'mean_hour': df.loc[mask, 'hour_of_day'].mean(),
            'std_hour': df.loc[mask, 'hour_of_day'].std() + 1e-5
        }
    
    entity_history = {}
    ip_failed_history = {}
    
    is_cold_start = []
    time_since_last_session = []
    distance_from_last = []
    implied_travel_speed = []
    
    failed_auth_entity_15m = []
    failed_auth_ip_15m_distinct = []
    
    is_new_resource = []
    new_resources_last_10 = []
    is_off_hours = []
    
    device_differs = []
    os_differs = []
    mac_differs = []
    firmware_differs = []
    
    session_duration_z = []
    
    for idx, row in df.iterrows():
        e_id = row['entity_id']
        e_type = row['entity_type']
        ts = row['timestamp']
        geo = row['geo_location']
        ip = row['source_ip']
        res = row['resource_accessed']
        cmd = row['command_sequence']
        device = row['device_fingerprint']
        dur = row['session_duration']
        
        is_failed = (cmd == "FAILED_AUTH")
        
        if e_id not in entity_history:
            entity_history[e_id] = {
                'sessions': 0,
                'last_ts': None,
                'last_geo': None,
                'failed_auths': [],
                'seen_resources': set(),
                'recent_new_res_list': [],
                'known_device': device,
                'durations': [],
                'login_hours': []
            }
        
        hist = entity_history[e_id]
        
        cold = (hist['sessions'] < 5)
        is_cold_start.append(cold)
        
        if hist['last_ts'] is not None:
            dt_hours = (ts - hist['last_ts']).total_seconds() / 3600.0
            time_since_last_session.append(dt_hours)
            dist = calculate_distance(geo, hist['last_geo'])
            distance_from_last.append(dist)
            speed = dist / max(dt_hours, 0.001)
            implied_travel_speed.append(speed)
        else:
            time_since_last_session.append(-1.0)
            distance_from_last.append(0.0)
            implied_travel_speed.append(0.0)
            
        if is_failed:
            hist['failed_auths'].append(ts)
        
        cutoff_15m = ts - timedelta(minutes=15)
        hist['failed_auths'] = [t for t in hist['failed_auths'] if t >= cutoff_15m]
        failed_auth_entity_15m.append(len(hist['failed_auths']))
        
        if ip not in ip_failed_history:
            ip_failed_history[ip] = []
        if is_failed:
            ip_failed_history[ip].append((ts, e_id))
            
        ip_failed_history[ip] = [(t, e) for (t, e) in ip_failed_history[ip] if t >= cutoff_15m]
        distinct_entities = len(set(e for (t, e) in ip_failed_history[ip]))
        failed_auth_ip_15m_distinct.append(distinct_entities)
        
        new_res = res not in hist['seen_resources']
        is_new_resource.append(1 if new_res else 0)
        
        hist['seen_resources'].add(res)
        hist['recent_new_res_list'].append(1 if new_res else 0)
        if len(hist['recent_new_res_list']) > 10:
            hist['recent_new_res_list'].pop(0)
        new_resources_last_10.append(sum(hist['recent_new_res_list']))
        
        known_dev = hist['known_device']
        if device != known_dev:
            device_differs.append(1)
            try:
                kd_parts = known_dev.split('|')
                d_parts = device.split('|')
                os_differs.append(1 if kd_parts[0] != d_parts[0] else 0)
                mac_differs.append(1 if kd_parts[1] != d_parts[1] else 0)
                firmware_differs.append(1 if kd_parts[2] != d_parts[2] else 0)
            except:
                os_differs.append(1)
                mac_differs.append(1)
                firmware_differs.append(1)
        else:
            device_differs.append(0)
            os_differs.append(0)
            mac_differs.append(0)
            firmware_differs.append(0)
            
        if not cold and len(hist['durations']) > 2:
            mean_dur = np.mean(hist['durations'])
            std_dur = np.std(hist['durations']) + 1e-5
        else:
            mean_dur = global_stats[e_type]['mean_dur']
            std_dur = global_stats[e_type]['std_dur']
        session_duration_z.append((dur - mean_dur) / std_dur)
        
        if not cold and len(hist['login_hours']) > 2:
            mean_hour = np.mean(hist['login_hours'])
            std_hour = np.std(hist['login_hours']) + 1e-5
        else:
            mean_hour = global_stats[e_type]['mean_hour']
            std_hour = global_stats[e_type]['std_hour']
            
        hour_diff = min(abs(row['hour_of_day'] - mean_hour), 24 - abs(row['hour_of_day'] - mean_hour))
        is_off_hours.append(1 if hour_diff > max(3, 2 * std_hour) else 0)
        
        hist['sessions'] += 1
        hist['last_ts'] = ts
        hist['last_geo'] = geo
        if dur > 0:
            hist['durations'].append(dur)
        hist['login_hours'].append(row['hour_of_day'])

    df['is_cold_start'] = is_cold_start
    df['time_since_last_session_hrs'] = time_since_last_session
    df['distance_from_last_km'] = distance_from_last
    df['implied_travel_speed_kmh'] = implied_travel_speed
    df['failed_auth_entity_15m'] = failed_auth_entity_15m
    df['failed_auth_ip_15m_distinct'] = failed_auth_ip_15m_distinct
    df['is_new_resource'] = is_new_resource
    df['new_resources_last_10'] = new_resources_last_10
    df['is_off_hours'] = is_off_hours
    df['device_differs'] = device_differs
    df['os_differs'] = os_differs
    df['mac_differs'] = mac_differs
    df['firmware_differs'] = firmware_differs
    df['session_duration_z'] = session_duration_z
    
    return df

def main():
    print("Loading full dataset...")
    df = pd.read_csv("full_labeled_dataset.csv")
    df_engineered = process_features(df)
    df_engineered.to_csv("engineered_features_labeled.csv", index=False)
    
    print("Loading inference dataset...")
    df_inf = pd.read_csv("inference_dataset.csv")
    df_inf_engineered = process_features(df_inf)
    df_inf_engineered.to_csv("engineered_features_inference.csv", index=False)
    
    print("Done. Saved to engineered_features_labeled.csv and engineered_features_inference.csv")
    
    print("Sanity check - engineered_features_labeled.csv shape:", df_engineered.shape)
    
if __name__ == "__main__":
    main()
