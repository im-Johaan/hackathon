# AI-Powered Behavioral Anomaly Detection for Cybersecurity
## Final Project Report

### 1. Assumptions
**Data Generation (Stage 1)**
- **Entities**: Simulated Users, Service Accounts, and Edge Devices with established baseline profiles (geo, IP, hours, resources, auth methods, and device fingerprints).
- **Normal Traffic**: Sampled within profile bounds; IP varies ~10% of the time to simulate dynamic networks.
- **Anomaly Injection Rate**: Injected anomalies uniformly across the session count at a fixed rate of ~3%.
- **Attack Simulations**: 
  - *Impossible Travel*: Consecutive logins spaced 5 minutes apart in distinct geographic regions with new IPs.
  - *Credential Stuffing*: Single attacking IP rapidly iterating across distinct entity IDs.
  - *Low-and-Slow Exfiltration*: Severely off-hours, high duration, explicit data export command sequences.
  - *(See `data_generation_assumptions.md` for full breakdown).*

**Feature Engineering & Modeling (Stages 2-4)**
- Engineered features assume availability of a historical state (rolling failed auths, seen resources).
- Distance metrics rely on a simplified coordinate mapping of geographic regions.
- Concept drift assumes that the recent stream of normal-classified traffic is safe to fine-tune on (which introduces a minor risk of data poisoning if an attacker stays below the anomaly threshold).

---

### 2. Methodology
Our solution implements a comprehensive, end-to-end pipeline:
**Synthetic Data Gen ➔ Feature Engineering ➔ GRU Autoencoder ➔ Interpretable Classification ➔ Streamlit Dashboard**

#### Addressing the Core Challenges
1. **Sequential Behavior**: We framed the problem as per-entity sequence modeling. Rather than evaluating isolated events, we build sliding windows of 10 sequential sessions. Our detection model (a GRU Autoencoder) processes these temporal sequences directly.
2. **Extreme Class Imbalance**: We utilized a semi-supervised setup. The GRU Autoencoder is trained *exclusively* on benign windows to learn "normal behavior reconstruction." It is never trained on rare attack examples, avoiding extreme class imbalance pitfalls. Anomaly score = reconstruction error.
3. **Concept Drift**: To prevent static profiles from decaying, the system streams new traffic and incrementally fine-tunes the GRU Autoencoder on recent benign sequences, adapting to new normal behaviors dynamically.
4. **Cold Start**: For entities with fewer than 5 historical sessions, the system gracefully falls back to population-level statistics (mean/std duration and login hours for their specific entity type) and uses a population-average baseline error augmented with specific heuristic penalties (e.g., new device + fast travel).
5. **Explainability**: While the GRU Autoencoder is powerful, it is a black box. We implemented an interpretable Decision Tree/Rule-based classifier that runs *only* on the top 1% flagged anomalous events. This layer uses interpretable engineered features to assign a specific attack category and generate a plain-English explanation (e.g., *"Flagged as brute force: 5 failed auths for this entity in 15m"*).

*Note: The GRU Autoencoder serves as both the baseline profiling representation (Deliverable 2) and the detection model (Deliverable 3). Combining these ensures an efficient, unified architecture rather than maintaining redundant profiling state machines.*

---

### 3. Evaluation Metrics
*Metrics are derived directly from the Stage 3 sequence model and Stage 4 classifier evaluation on the held-out labeled dataset.*

**Sequence Anomaly Detector (GRU Autoencoder)**
- **PR-AUC**: 0.4583
- **Top 1% Alert Budget Threshold**: 1.7476
- **Precision at Top 1% Alert Budget**: 98.0% (98 of 100 alerts were true attacks)
- **Recall at Top 1% Alert Budget**: 18.0% (Caught the most egregious sequence deviations first)

**Alert Classification & Explainability (Stage 4)**
The rule-based decision tree was applied to the flagged top 1% alerts to assign human-readable categories.
- **Overall Accuracy**: 98%
- **Credential Stuffing**: 100% Precision | 100% Recall (112/112 correctly classified)
- **Lateral Movement**: 86% Precision | 100% Recall (6/6 correctly classified)
- **False Positives**: The 2 benign events erroneously flagged by the GRU Autoencoder were forced into attack buckets by the rules (Impossible Travel/Lateral Movement).
- *Note: `insider_drift` was intentionally handled as an ambiguous edge-case for false-positive tuning and not reported as a confirmed attack.*

---

### 4. Known Limitations
- **Hand-Tuned Rules**: The rule-based classifier and decision tree were trained/tuned on synthetic data. These thresholds will likely require recalibration on real-world SOC traffic.
- **Simplified Concept Drift**: The concept-drift handling uses a batch rolling-window approach to fine-tune the model. A robust production system would require a more mature online-learning or continual-learning framework.
- **Synthetic Limitations**: Synthetic data, while adhering to specified patterns, cannot fully capture the adaptivity and noise of real-world attackers.
- **Feature Extraction**: Distance calculations use rough centroid mappings rather than precise IP-to-Geo latency measurements.

---

### 5. Future Work (Deferred Capabilities)
To meet hackathon scope and time constraints while prioritizing the core pipeline, we made deliberate engineering tradeoffs. Future iterations of this system should include:
- **Graph-Based Modeling**: Incorporating Graph Neural Networks (GNNs) to map entity-resource relationships for more sophisticated lateral movement detection.
- **Full Online Learning**: Upgrading the concept drift mechanism to a robust continual learning pipeline with safeguard mechanisms against data poisoning.
- **Richer Command Sequence Modeling**: Moving beyond simple strings to NLP-based embeddings for complex terminal or API command sequences.
