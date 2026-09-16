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

### Summary of Reviewed Work

| # | Study / Approach | Strengths | Limitations |
|---|---|---|---|
| 1 | Traditional ML Classifiers (e.g., RF, SVM) | Achieve high accuracy — up to 98.92% reported on UNSW-NB15 | Suffer from a higher False Alarm Rate (FAR) relative to artificial neural network (ANN)-based approaches |
| 2 | MLP combined with XGBoost-based Feature Selection & SMOTE | Effectively mitigates class imbalance in the training data | Depends on an external, separately-tuned feature selection pipeline (XGBoost), adding complexity |
| 3 | Comparative studies of Deep Learning models for NIDS | Shows that deep architectures such as DNN and LSTM extract richer feature representations from raw traffic than shallow learners, improving accuracy | Popular benchmark datasets (e.g., KDD Cup) contain large numbers of redundant records, which bias models toward frequently occurring attacks and hurt performance on rarer, more dangerous ones |
| 4 | CICIDS2017 Dataset Paper (Sharafaldin et al.) | Introduces a modern, realistic, and diverse dataset covering both benign traffic and multiple contemporary attack types (DoS, DDoS, brute force, infiltration, botnet, etc.), with rich network-flow-level features suitable for ML | The dataset is large and computationally expensive to process, and still exhibits significant class imbalance for rare attack categories |
| 5 | AdamW — Decoupled Weight Decay Regularization | Decouples weight decay from gradient updates, offering better regularization than standard L2 regularization used with Adam; helps reduce overfitting and is broadly applicable to CNNs, LSTMs, and other deep models | Addresses only the optimization/regularization aspect of training — it does not solve dataset imbalance, feature quality, or the intrusion-detection problem itself |
Broader Observations from the Literature

Beyond the individual studies summarized above, several recurring themes emerged across the wider body of research on ML/DL and GAN-based intrusion detection:

- **Traditional ML classifiers** (Random Forest, SVM, Logistic Regression) consistently achieve strong raw accuracy figures but tend to generate more false alarms than neural-network-based alternatives, which is problematic in operational settings where alert fatigue is a real concern.
- **Deep learning models**, particularly DNNs and LSTMs, are better at automatically extracting meaningful features from raw or lightly processed traffic data, with some studies reporting accuracy as high as 99.4% on the NSL-KDD dataset.
- **Unsupervised anomaly-detection approaches** offer a genuine advantage in detecting zero-day attacks (since they do not rely on labeled attack examples), but this comes at the cost of a considerably higher false-positive rate.
- **Legacy benchmark datasets** such as KDD Cup are widely criticized for containing large numbers of duplicate or redundant records, which skew model training toward the most common attacks while leaving rarer, high-impact attacks poorly represented and poorly classified.
- **Class imbalance** is arguably the single most persistent challenge across the field — minority classes such as Web attacks and Heartbleed are frequently misclassified regardless of the underlying model architecture.
### Identified Research Gap

Synthesizing the above findings, the literature review pointed toward a clear gap: there is a need for a **hybrid class-balancing strategy** — combining classical oversampling (SMOTE) with generative approaches (WGAN-GP) — paired with an **ensemble classification strategy**, in order to achieve robust detection performance across both common and rare attack categories simultaneously. This gap directly informed the design of the solution proposed in this project.

## Methodology

### Overview of the Proposed Pipeline

To address the gaps identified above, this project proposes a complete, **end-to-end, software-only ML-based NIDS pipeline**. Being entirely software-based, the solution avoids the cost and deployment overhead of dedicated hardware security appliances, making it a practical option for organizations of varying sizes.

The pipeline consists of the following major stages:

1. **Data acquisition and preprocessing** of the CICIDS2017 dataset.
2. **Feature engineering**, using flow-level statistics generated by CICFlowMeter.
3. **Class balancing**, using SMOTE, ADASYN, and WGAN-GP to correct the severe imbalance between benign and attack classes.
4. **Model training**, covering both traditional ML classifiers and deep learning architectures.
5. **Ensemble aggregation**, using soft-voting across the trained deep learning models.
6. **Performance evaluation**, using standard classification metrics.

### Dataset Description

The project uses **CICIDS2017**, a widely-used, modern benchmark dataset created specifically to overcome the shortcomings of older datasets like KDD Cup.

| Property | Details |
|---|---|
| Dataset Name | CICIDS2017 |
| Total Records | ≈ 2.8 million network flow records |
| Total Features | 80 network flow features |
| Classes | 15 distinct attack types, plus a Benign class |

The dataset captures a diverse mix of realistic attack behavior, including DoS, DDoS, brute-force, infiltration, and botnet traffic, alongside normal (benign) traffic, making it well suited for training a generalizable NIDS model.

### Data Preprocessing and Feature Engineering

Raw network captures are not directly usable for model training. The following preprocessing steps were applied:

- **Data cleaning** — removing corrupted, duplicate, or missing entries.
- **Normalization** — scaling numerical feature values into a consistent range so that no single feature dominates model training due to its magnitude.
- **Feature engineering** — using 76 flow-level statistics generated via **CICFlowMeter**, which converts raw packet captures into structured, per-flow numerical features (e.g., flow duration, packet counts, byte counts, inter-arrival times) suitable for machine learning.

  <img width="290" height="751" alt="image" src="https://github.com/user-attachments/assets/2fe1f76f-c878-4b5e-b44d-b782dc5debe0" />


### Addressing Class Imbalance

One of the central challenges tackled in this project — as highlighted repeatedly in the literature review — is the severe imbalance between benign traffic and various attack categories (with some attack types, such as Web attacks and Heartbleed, being extremely rare). To correct this, three complementary techniques were applied:

- **SMOTE (Synthetic Minority Over-sampling Technique):** Generates new synthetic minority-class samples by interpolating between existing minority-class examples.
- **ADASYN (Adaptive Synthetic Sampling):** An adaptive variant of SMOTE that generates more synthetic samples for minority-class examples that are harder to classify.
- **WGAN-GP (Wasserstein GAN with Gradient Penalty):** A generative adversarial network used to produce higher-quality, more realistic synthetic samples for the most severely underrepresented classes, going beyond what interpolation-based methods like SMOTE and ADASYN can achieve alone.

Using these three techniques together allows the resulting training data to be far more balanced, which in turn allows the trained models to learn meaningful decision boundaries for rare attack types instead of being overwhelmed by the dominant benign class.

<img width="1009" height="349" alt="image" src="https://github.com/user-attachments/assets/f6a4dec7-a440-4118-a0e5-a952cc9cb834" />


### Model Training Strategy

Two broad categories of classifiers were trained and compared:

- **Traditional Machine Learning Classifiers:** Random Forest, Decision Tree, and Support Vector Machine (SVM) — chosen as strong, interpretable baselines consistent with those seen in the literature review.
- **Deep Learning Classifiers:** CNN-1D, BiLSTM, and MLP — chosen specifically to capture complementary types of patterns in the traffic data (local patterns, sequential dependencies, and complex nonlinear relationships, respectively).

### Ensemble Strategy

Rather than relying on a single deep learning model, this project combines the predictions of CNN-1D, BiLSTM, and MLP using a **soft-voting ensemble**, where the probability outputs of all three models are averaged (or weighted) to produce a single, more robust final prediction. This approach reduces the impact of any individual model's weaknesses and generally yields more stable performance across diverse attack types.


### Evaluation Metrics

Model performance was assessed using standard classification metrics, including:

- **Accuracy** — overall proportion of correctly classified samples.
- **Precision** — proportion of predicted attacks that are actually attacks (i.e., how trustworthy a positive prediction is).
- **Recall** — proportion of actual attacks that were successfully detected (i.e., how many real attacks the model catches).
- **F1-score** — the harmonic mean of precision and recall, useful for imbalanced classification problems.
- **Confusion matrix analysis** — to examine per-class misclassification patterns in detail.

---

## System Design and Model Architecture

### Ensemble Model Architecture

The core detection engine is built around three complementary deep learning components, combined into a single ensemble:

1. **Input Layer:** Accepts network flow features extracted from the CICIDS2017 dataset (or, during live detection, from real-time captured traffic).
2. **Preprocessing Layer:** Performs data cleaning, normalization, and formatting consistent with the training pipeline.
3. **1D-CNN Branch:** Captures local, short-range patterns within the traffic feature sequence.
4. **BiLSTM Branch:** Learns sequential and bidirectional dependencies across the flow features, capturing temporal relationships that a purely convolutional or feed-forward model might miss.
5. **MLP Branch:** Models complex, nonlinear relationships between features that are not necessarily sequential or local in nature.
6. **Soft-Voting Ensemble Layer:** Aggregates the outputs of the CNN-1D, BiLSTM, and MLP branches to produce a single, more reliable prediction.
7. **Output Layer:** Produces the final classification — either **BENIGN** or a specific **ATTACK** category.

<img width="518" height="760" alt="image" src="https://github.com/user-attachments/assets/bb21ec11-3f39-46e5-8986-47524637244e" />

### System Operation: Training vs. Live Detection

The overall system is designed to operate in two distinct phases:

- **Training Phase:** The CICIDS2017 dataset is preprocessed, balanced using SMOTE/ADASYN/WGAN-GP, and used to train the CNN-1D, BiLSTM, and MLP models independently. Their outputs are then combined via soft-voting to form the final ensemble classifier, which is saved for later use.
- **Live Detection Phase:** Real-world network traffic is captured, converted into the same flow-based feature format used during training, and passed through the trained ensemble model to classify traffic as benign or malicious in real time, with results surfaced to the user through the monitoring dashboard.
# System Model

<img width="1081" height="526" alt="image" src="https://github.com/user-attachments/assets/dfe6ef54-bf6b-4be9-9f57-21a2b11cfb85" />

---

## Implementation and Deployment

### Attack Simulation ("Run-Attack") Pipeline

To rigorously test the system under realistic conditions, an attack-traffic generation pipeline was built to simulate a variety of attack types — including DoS, DDoS, brute-force, and botnet traffic — allowing the detection system to be validated against controlled, reproducible attack scenarios rather than relying solely on static dataset evaluation.

<img width="538" height="720" alt="image" src="https://github.com/user-attachments/assets/21baf7f3-008a-4531-a70c-6b2ad01321ba" />


### Real-Time Detection Pipeline

The live detection pipeline is responsible for:

1. Capturing network traffic in real time (using packet-capture tooling built with **Scapy**).
2. Extracting flow-based features that match the schema used during model training (i.e., consistent with the CICIDS2017 / CICFlowMeter feature set).
3. Feeding these features into the trained ensemble model for immediate classification.
4. Forwarding the classification results to the monitoring dashboard for visualization.

<img width="538" height="711" alt="image" src="https://github.com/user-attachments/assets/2aa6670e-350e-4e81-96c2-0fb608bc2c1a" />


### Monitoring Dashboard

A web-based monitoring dashboard was developed using **Flask** to give administrators a clear, accessible view of ongoing detection activity. The dashboard displays live classification counts by category, allowing at-a-glance situational awareness of network health.

**Sample output captured during a real-time detection run:**

| Category | Predicted Count | Description |
|---|---|---|
| DoS | 135 | Denial-of-Service attack attempts |
| Infiltration | 106 | Unauthorized access attempts |
| BENIGN | 42 | Normal traffic |
| DDoS | 14 | Distributed Denial-of-Service |
| Brute Force | 9 | Password-guessing attacks |
| Bot | 2 | Botnet-related traffic |
| **Total Predictions** | **318** | |

<img width="741" height="619" alt="image" src="https://github.com/user-attachments/assets/67274967-1a64-4612-8a49-3209572ba767" />


---

## Experimental Results and Evaluation

### Traditional Machine Learning Results

| Model | Accuracy |
|---|---|
| SVM | 95.92% |
| Decision Tree | 98.51% |
| Random Forest | 98.92% |

Among the traditional classifiers, **Random Forest** performed the best overall, closely followed by Decision Tree, both comfortably outperforming SVM. These results are broadly consistent with figures reported in the literature review for similar classifiers on comparable intrusion-detection datasets.

### Deep Learning Ensemble Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| BENIGN | 0.997 | 0.849 | 0.917 |
| DDoS | 0.025 | 0.972 | 0.048 |
| DoS | 0.524 | 0.970 | 0.680 |

### Discussion and Analysis

Several important observations can be drawn from the results:

- The **BENIGN class** is classified with very high precision (0.997), meaning the model rarely mislabels normal traffic as an attack — an important property for minimizing unnecessary alerts.
- Both **DDoS and DoS classes** show very high recall (≈0.97), which means the ensemble successfully catches the overwhelming majority of actual attacks in these categories — a strong result from a security standpoint, since missed attacks are typically far more costly than false alarms.
- However, the **precision for the DDoS class is notably low (0.025)**, indicating that a large number of the model's DDoS predictions are false positives. This is a direct manifestation of the class imbalance challenge discussed in Chapters 1 and 2 — even after applying SMOTE, ADASYN, and WGAN-GP, some residual imbalance and feature overlap between classes remains, causing the model to over-predict the minority DDoS class.
- Taken together, these results validate the project's core hypothesis — that an ML/DL ensemble approach can achieve strong detection coverage — while also confirming that class imbalance remains a genuinely difficult, only partially solved problem, pointing to a clear direction for future refinement (discussed further in Chapter 9).

---

## Impact Assessment

### Societal Impact

The project contributes to improved digital safety at both the individual and organizational level by enabling:

- Improved overall network security posture.
- Early detection of threats before they can cause significant damage.
- Stronger protection against cyberattacks for the individuals and organizations relying on the protected network.

### Environmental Impact

Because the proposed NIDS is entirely software-based:

- It does not require additional specialized hardware, reducing manufacturing and energy overhead.
- It can be deployed on existing computing infrastructure, maximizing the utility of hardware organizations already own.
- It indirectly contributes to reduced electronic waste compared to hardware-based security appliances that eventually require replacement.

### Lifelong Learning Outcomes

Throughout the course of this project, the team gained hands-on, industry-relevant experience with a range of tools, languages, and concepts, including:

- Machine learning and deep learning theory and practice (ML/DL)
- Python programming for data science and model development
- Core network protocols and traffic analysis concepts
- **Scapy** for packet capture and manipulation
- **Flask** for building web-based dashboards and APIs
- Broader cybersecurity principles and practical intrusion-detection concepts

These skills are directly transferable to both further academic research and professional roles in cybersecurity, data science, and software engineering.

---



## Conclusion and Future Work

### Conclusion

This project successfully designed, implemented, and evaluated a Machine Learning-based Network Intrusion Detection System built on the CICIDS2017 dataset. The key achievements of the project include:

- Development of a complete, end-to-end ML-based NIDS pipeline, from raw data to real-time detection.
- Evaluation and comparison of multiple traditional ML classifiers (SVM, Decision Tree, Random Forest).
- Identification and analysis of class imbalance as a central, recurring challenge in intrusion detection.
- Implementation of SMOTE and WGAN-GP-based techniques to mitigate class imbalance during model development.
- Design of a CNN-1D + BiLSTM + MLP deep learning ensemble, combined via soft-voting.
- Development of a functioning real-time traffic detection module.
- Integration of a live monitoring dashboard for visualizing detection results.

Collectively, these contributions demonstrate that a fully software-based ML/DL approach can serve as an effective, scalable, and cost-efficient alternative to traditional signature-based intrusion detection systems — while also making clear that class imbalance remains an open challenge that continues to affect precision on certain attack categories, even with modern balancing techniques applied.

### Future Work

Building on the current results, several directions could further improve the system:

- Further tuning of the WGAN-GP architecture and ensemble weighting to specifically improve precision on minority classes such as DDoS.
- Extending the system to additional or more recent intrusion-detection datasets to test generalizability beyond CICIDS2017.
- Exploring additional deep learning architectures (e.g., Transformer-based models) for potentially richer feature representation.
- Deploying the system in a live production or campus-network environment for longer-term, real-world validation.

---

## References

1. I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," in *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116.
2. A. H. Lashkari, G. Draper-Gil, M. S. I. Mamun, and A. A. Ghorbani, "Characterization of Tor Traffic using Time Based Features," in *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 2017.
3. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.
4. H. He, Y. Bai, E. A. Garcia, and S. Li, "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning," in *Proceedings of the IEEE International Joint Conference on Neural Networks*, 2008, pp. 1322–1328.
5. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," *arXiv preprint arXiv:1711.05101*, 2017.
6. A. Smith, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates," in *Proceedings of the SPIE Defense + Commercial Sensing*, 2018.
7. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
8. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

---



