import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI Anomaly Detection", layout="wide")

@st.cache_data
def load_data():
    alerts_df = pd.read_csv("scored_alerts.csv")
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    
    sessions_df = pd.read_csv("session_scores.csv")
    sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'])
    
    entity_mapping = sessions_df[['entity_id', 'entity_type']].drop_duplicates().set_index('entity_id')['entity_type'].to_dict()
    alerts_df['entity_type'] = alerts_df['entity_id'].map(entity_mapping)
    
    return alerts_df, sessions_df

alerts_df, sessions_df = load_data()

st.title("🛡️ AI-Powered Behavioral Anomaly Detection")

with st.expander("ℹ️ About / Methodology"):
    st.markdown("""
    **Pipeline Stages:**
    1. **Data Generation**: Synthetic access logs per entity profiling benign and 7 specific attack patterns.
    2. **Feature Engineering**: Sequences transformed into interpretable temporal, geo-velocity, auth, and resource features.
    3. **Sequence Anomaly Detection (GRU Autoencoder)**: Trained purely on benign sequences to reconstruct expected behavior. High reconstruction error = Anomaly Score. Handles cold starts via population fallback and concept drift via incremental recent updates.
    4. **Rule-Based Classification**: Interpretable logic layers on top of the Top 1% flagged anomalies to categorize the specific attack type (e.g. Impossible Travel, Credential Stuffing) and provide a human-readable explanation.
    """)

# Summary View
st.header("📊 Threat Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Alerts (Top 1% Budget)", len(alerts_df))
with col2:
    threshold = alerts_df['anomaly_score'].min() if len(alerts_df) > 0 else 0
    st.metric("Current Alert Threshold", f"{threshold:.4f}")
with col3:
    critical_alerts = len(alerts_df[alerts_df['predicted_attack_type'] != 'insider_drift'])
    st.metric("Confirmed Attack Patterns", critical_alerts)

col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    attack_counts = alerts_df['predicted_attack_type'].value_counts().reset_index()
    attack_counts.columns = ['Attack Type', 'Count']
    fig1 = px.bar(attack_counts, x='Attack Type', y='Count', title="Alerts by Attack Type")
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    alerts_by_date = alerts_df.set_index('timestamp').resample('D').size().reset_index(name='Count')
    fig2 = px.line(alerts_by_date, x='timestamp', y='Count', title="Alert Volume Over Time")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Filters
st.header("🚨 Ranked Alert Queue")
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    attack_types = ['All'] + list(alerts_df['predicted_attack_type'].unique())
    selected_attack = st.selectbox("Filter by Attack Type", attack_types)
with filter_col2:
    entity_types = ['All'] + list(alerts_df['entity_type'].dropna().unique())
    selected_entity = st.selectbox("Filter by Entity Type", entity_types)
with filter_col3:
    min_date = alerts_df['timestamp'].min().date() if len(alerts_df) > 0 else datetime.today().date()
    max_date = alerts_df['timestamp'].max().date() if len(alerts_df) > 0 else datetime.today().date()
    date_range = st.date_input("Date Range", [min_date, max_date])

filtered_df = alerts_df.copy()
if selected_attack != 'All':
    filtered_df = filtered_df[filtered_df['predicted_attack_type'] == selected_attack]
if selected_entity != 'All':
    filtered_df = filtered_df[filtered_df['entity_type'] == selected_entity]
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= date_range[0]) & (filtered_df['timestamp'].dt.date <= date_range[1])]

filtered_df = filtered_df.sort_values('anomaly_score', ascending=False)

# Display table
st.dataframe(filtered_df[['timestamp', 'entity_id', 'entity_type', 'predicted_attack_type', 'anomaly_score', 'explanation_text']], use_container_width=True)

st.divider()

st.header("🔍 Alert Detail View")
if not filtered_df.empty:
    selected_event_id = st.selectbox("Select an Event ID to view details:", filtered_df['event_id'])
    
    event = filtered_df[filtered_df['event_id'] == selected_event_id].iloc[0]
    
    st.subheader(f"Alert Details for {event['entity_id']} at {event['timestamp']}")
    st.info(event['explanation_text'])
    
    # Flags
    flags = []
    if event['is_cold_start'] == 1:
        flags.append("❄️ **Cold Start Entity** (Population stats used)")
    else:
        flags.append("🔄 **Drift-Updated Profile** (Entity has established history)")
        
    for flag in flags:
        st.markdown(flag)
        
    col_feat, col_hist = st.columns([1, 2])
    
    with col_feat:
        st.write("**Contributing Feature Values**")
        feature_cols = [
            'time_since_last_session_hrs', 'distance_from_last_km', 
            'implied_travel_speed_kmh', 'failed_auth_entity_15m', 
            'failed_auth_ip_15m_distinct', 'is_new_resource', 
            'new_resources_last_10', 'is_off_hours', 'device_differs', 
            'session_duration_z'
        ]
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Value': [event[c] if c in event else 'N/A' for c in feature_cols]})
        st.dataframe(feat_df, use_container_width=True)
        
    with col_hist:
        st.write("**Recent Session History (Last 20 Sessions)**")
        entity_history = sessions_df[sessions_df['entity_id'] == event['entity_id']].sort_values('timestamp')
        # Get up to this event
        entity_history = entity_history[entity_history['timestamp'] <= event['timestamp']].tail(20)
        
        hist_cols = ['timestamp', 'geo_location', 'resource_accessed', 'anomaly_score']
        # Also include command_sequence and auth_method for more context if available
        if 'command_sequence' in entity_history.columns:
            hist_cols.insert(2, 'command_sequence')
            
        st.dataframe(entity_history[hist_cols].style.highlight_max(subset=['anomaly_score'], color='red'), use_container_width=True)
else:
    st.write("No alerts match the current filters.")
