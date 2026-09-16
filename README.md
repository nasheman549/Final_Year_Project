# ML Based Network Intrusion Detection System Using Machine learning
## Introduction
## Intrusion Detection System (IDS)
Network security has become one of the most pressing concerns in the digital era, as organizations and individuals rely heavily on interconnected systems for communication, data storage, and business operations. An **Intrusion Detection System (IDS)** is a security mechanism built specifically to observe, analyze, and respond to activity occurring within a computer system or a network. Its core responsibilities include:
- Continuously monitoring system or network behavior for anomalies.
- Recognizing patterns associated with unauthorized access, misuse, or malicious intent.
- Raising timely alerts so that administrators can respond before damage occurs.
Depending on where the monitoring takes place, IDS solutions are broadly categorized into two types:
- **Host-Based Intrusion Detection System (HIDS):** Deployed on an individual machine, this type inspects local activity such as system calls, running processes, and log files to catch suspicious behavior on that specific host.
- **Network-Based Intrusion Detection System (NIDS):** Deployed at strategic points within a network, this type inspects traffic flowing across the network to detect malicious activity affecting multiple devices simultaneously.
This project is concerned specifically with the design and implementation of a **Network-Based IDS**, since network-level monitoring provides broader visibility and is more suitable for detecting large-scale or distributed attacks.
## Limitations of Conventional NIDS
The majority of NIDS solutions in use today rely on **signature-based detection**, a method in which the system compares incoming traffic against a stored database of known attack patterns. While this approach performs reliably against previously catalogued threats, it comes with a fundamental weakness: it is entirely reactive. A signature-based system can only recognize an attack once that attack's pattern has already been documented and added to its database. This creates a dangerous blind spot against:
- **Zero-day attacks** — new attack types that have never been seen or documented before.
- **Evolving or obfuscated attack variants** — slightly modified versions of known attacks that no longer match the stored signature exactly.
Given the pace at which attackers are innovating, this reactive model is no longer sufficient to defend modern networks.
Rationale for a Machine Learning Approach

The above limitations motivate the use of **Machine Learning (ML)** for intrusion detection. Rather than relying on static, pre-defined rules, an ML-based NIDS learns the underlying statistical patterns that separate normal ("benign") traffic from malicious traffic. Once trained, such a model can generalize this understanding to flag traffic it has never explicitly seen before — including many zero-day-style attacks — because it is reasoning about *behavioral patterns* rather than *exact signatures*.

Several converging factors make this an urgent and worthwhile problem to solve:

- Cyber-attacks are increasing continuously in **volume, diversity, and sophistication**, straining traditional defenses.
- **Manual inspection** of network traffic by human analysts is slow, labor-intensive, and simply does not scale to modern data rates.
- Machine learning models can be trained to **automatically detect anomalies** in traffic behavior without constant human intervention.
- Real-world network data tends to be **high-dimensional, noisy, and heavily imbalanced** (i.e., normal traffic vastly outnumbers attack traffic), which demands careful data engineering before any model can learn from it effectively.
- A **purely software-based detection system** is attractive because it can be deployed cost-effectively on existing infrastructure, without the need for specialized security appliances.
Project Objectives

Building on this rationale, the specific objectives pursued in this project are to:

1. Design and build a complete, end-to-end ML/DL-based NIDS pipeline using a modern, realistic benchmark dataset.
2. Address the class imbalance problem inherent in intrusion detection datasets through resampling and synthetic data generation techniques.
3. Develop and compare both traditional machine learning classifiers and deep learning architectures for the classification task.
4. Combine multiple models into an ensemble to improve robustness and overall detection performance.
5. Build a real-time detection pipeline capable of classifying live network traffic.
6. Provide a dashboard interface through which detection results can be visualized and monitored in real time.
Related Work and Research Gap

Before designing our solution, an extensive review of existing academic literature on ML/DL-based and GAN-based intrusion detection was carried out. The purpose of this review was to understand what has already been tried, what worked well, and — more importantly — where the existing gaps lie that this project could address.

### 2.1 Summary of Reviewed Work

| # | Study / Approach | Strengths | Limitations |
|---|---|---|---|
| 1 | Traditional ML Classifiers (e.g., RF, SVM) | Achieve high accuracy — up to 98.92% reported on UNSW-NB15 | Suffer from a higher False Alarm Rate (FAR) relative to artificial neural network (ANN)-based approaches |
| 2 | MLP combined with XGBoost-based Feature Selection & SMOTE | Effectively mitigates class imbalance in the training data | Depends on an external, separately-tuned feature selection pipeline (XGBoost), adding complexity |
| 3 | Comparative studies of Deep Learning models for NIDS | Shows that deep architectures such as DNN and LSTM extract richer feature representations from raw traffic than shallow learners, improving accuracy | Popular benchmark datasets (e.g., KDD Cup) contain large numbers of redundant records, which bias models toward frequently occurring attacks and hurt performance on rarer, more dangerous ones |
| 4 | CICIDS2017 Dataset Paper (Sharafaldin et al.) | Introduces a modern, realistic, and diverse dataset covering both benign traffic and multiple contemporary attack types (DoS, DDoS, brute force, infiltration, botnet, etc.), with rich network-flow-level features suitable for ML | The dataset is large and computationally expensive to process, and still exhibits significant class imbalance for rare attack categories |
| 5 | AdamW — Decoupled Weight Decay Regularization | Decouples weight decay from gradient updates, offering better regularization than standard L2 regularization used with Adam; helps reduce overfitting and is broadly applicable to CNNs, LSTMs, and other deep models | Addresses only the optimization/regularization aspect of training — it does not solve dataset imbalance, feature quality, or the intrusion-detection problem itself |
