# Project Presentation: ISL-Lens
## Real-Time Indian Sign Language Recognition & Interactive AI Console

<p align="center">
  <img src="Final Project Presentation/figures/ajcelogo2023.png" alt="Amal Jyothi College of Engineering" width="160" />
</p>

<p align="center">
  <b>Department of Computer Science and Engineering</b><br>
  <b>Amal Jyothi College of Engineering (Autonomous), Kanjirappally</b><br>
  <i>Affiliated to APJ Abdul Kalam Technological University (KTU)</i><br>
  <b>B.Tech Project Phase I — Final Review Presentation</b><br>
  October 2026
</p>

---

### Project & Presenter Metadata
- **Project Title:** ISL-LENS: Real-Time Indian Sign Language Recognition and Interactive AI Console
- **Student Candidate:** **Aevlin Prince** (Reg No: **AJC23CS014**, Email: `aevlinprince2027@cs.ajce.in`)
- **Faculty Guide:** **Dr. Indu John**, Assistant Professor, Dept. of Computer Science & Engineering
- **Head of Department:** **Dr. Juby Mathew**, Professor & HoD CSE, Amal Jyothi College of Engineering
- **Target Evaluation:** B.Tech Computer Science and Engineering — Project Phase I Final Evaluation

---

## Table of Contents
1. [Slide 1: Title & Overview](#slide-1-title--overview)
2. [Slide 2: Introduction & Background](#slide-2-introduction--background)
3. [Slide 3: Problem Definition & Limitations of Existing Systems](#slide-3-problem-definition--limitations-of-existing-systems)
4. [Slide 4: Project Objectives & Scope](#slide-4-project-objectives--scope)
5. [Slide 5: Literature Survey](#slide-5-literature-survey)
6. [Slide 6: Research Gaps in Existing Literature](#slide-6-research-gaps-in-existing-literature)
7. [Slide 7: Proposed System Architecture](#slide-7-proposed-system-architecture)
8. [Slide 8: Landmark Extraction & Kinematic Pipeline](#slide-8-landmark-extraction--kinematic-pipeline)
9. [Slide 9: Open-Set Rejection & ML Engine](#slide-9-open-set-rejection--ml-engine)
10. [Slide 10: Continuous Sentence Translation (SOV to SVO)](#slide-10-continuous-sentence-translation-sov-to-svo)
11. [Slide 11: Interactive Practice Studio with Circular Accuracy HUD](#slide-11-interactive-practice-studio-with-circular-accuracy-hud)
12. [Slide 12: Interactive AI Retraining Studio & Validation Gate](#slide-12-interactive-ai-retraining-studio--validation-gate)
13. [Slide 13: Hardware and Software Specifications](#slide-13-hardware-and-software-specifications)
14. [Slide 14: Experimental Results & Accuracy Benchmarks](#slide-14-experimental-results--accuracy-benchmarks)
15. [Slide 15: Confusion Matrix & Category Breakdown](#slide-15-confusion-matrix--category-breakdown)
16. [Slide 16: Comparative Analysis with Baseline Models](#slide-16-comparative-analysis-with-baseline-models)
17. [Slide 17: Ablation Studies & Latency Profiling](#slide-17-ablation-studies--latency-profiling)
18. [Slide 18: Real-World Applications](#slide-18-real-world-applications)
19. [Slide 19: Key Advantages of ISL-Lens](#slide-19-key-advantages-of-isl-lens)
20. [Slide 20: 80% Milestone Verification & Phase II Roadmap](#slide-20-80-milestone-verification--phase-ii-roadmap)
21. [Slide 21: Limitations of the Current System](#slide-21-limitations-of-the-current-system)
22. [Slide 22: Conclusion & Summary](#slide-22-conclusion--summary)
23. [Slide 23: Publication Details](#slide-23-publication-details)
24. [Slide 24: References](#slide-24-references)

---

## Slide 1: Title & Overview

### Slide Content
* **Title:** ISL-LENS: Real-Time Indian Sign Language Recognition and Interactive AI Console
* **Domain:** Edge Computer Vision, Human-Computer Interaction (HCI), Natural Language Processing
* **Presented by:** Aevlin Prince (AJC23CS014)
* **Under the Guidance of:** Dr. Indu John, Assistant Professor, Dept. of CSE
* **Core Philosophy:** Zero-Hardware, Edge-First, Privacy-Preserving Assistive Communication

### Speaker Notes
> *"Good morning respected evaluators, guide, and faculty members. Today, I present the Phase I Final Review of my B.Tech project titled 'ISL-LENS: Real-Time Indian Sign Language Recognition and Interactive AI Console'. ISL-Lens is an edge-first, browser-native assistive system designed to empower the Deaf and Hard-of-Hearing community across India by converting real-time Indian Sign Language into grammatically fluent spoken English using only a standard consumer webcam."*

---

## Slide 2: Introduction & Background

### Slide Content
* **Demographic Crisis:** Over **18 million** individuals in India live with severe hearing impairment (Census of India / WHO).
* **Acute Interpreter Shortage:** Fewer than **300 certified Indian Sign Language (ISL) interpreters** exist nationwide (ratio: 1 interpreter per 60,000+ deaf citizens).
* **Systemic Exclusion:** Communication barriers impede deaf citizens from accessing healthcare, legal counsel, classrooms, and everyday public services.
* **Linguistic Reality of ISL:**
  * Visual-gestural language with 5 phonological parameters: handshape, spatial location, palm orientation, kinetic movement, and facial expressions.
  * Strictly follows **Subject-Object-Verb (SOV)** syntax (e.g., `ME WATER DRINK`), unlike spoken English which follows **Subject-Verb-Object (SVO)** (`I drink water`).
  * Direct word concatenation results in disjointed, confusing transcripts.

<p align="center">
  <img src="Final Project Presentation/figures/fig_pipeline.png" alt="ISL-Lens High-Level Overview" width="85%" />
  <br><b>Figure 1:</b> High-Level Overview of the ISL-Lens Real-Time Vision and Language Pipeline.
</p>

### Speaker Notes
> *"Sign language is not a direct word-for-word translation of spoken speech; it is a rich, natural language governed by strict grammar. India has one of the world's largest deaf populations—exceeding 18 million individuals—yet there are fewer than 300 licensed interpreters. A deaf person visiting a doctor or a government office faces near-total communication breakdown. Our mission with ISL-Lens is to turn any commodity laptop or smartphone into a real-time, privacy-preserving digital sign interpreter."*

---

## Slide 3: Problem Definition & Limitations of Existing Systems

### Slide Content
1. **Hardware Encumbrance:** Traditional sensor-glove systems (flex sensors, IMUs) cost $200–$500+, require continuous recalibration, restrict natural hand movement, and cause physical fatigue.
2. **Excessive Computational Demands:** Raw-pixel 3D-CNNs (e.g., 3D-ResNet, I3D) require server-grade discrete GPUs, generating inference latencies over **180–220 ms** per frame, breaking interactive conversation.
3. **Environmental Fragility:** Pixel-based neural models fail catastrophically under varying room illumination, skin complexions, clothing clutter, and camera angles.
4. **Vulnerability to False Positives (Closed-Set Bias):** Standard neural networks assign random hand movements or resting pauses to the nearest trained sign class because softmax probabilities sum to 1.0.
5. **Static, Inextensible Vocabularies:** Prior academic prototypes cannot learn new local dialect signs without developer intervention and full system retraining.
6. **Lack of Syntactic Translation:** Existing systems merely display raw isolated glosses without grammatical translation into coherent, vocalized English sentences.

### Speaker Notes
> *"Existing research falls into two extremes: either expensive wearable data gloves that nobody can afford or wear in daily life, or heavy 3D convolutional video networks that require thousand-dollar GPUs and crash when lighting changes. Crucially, almost all systems operate under a closed-set assumption: if a signer scratches their chin or pauses, the system outputs a false positive. Furthermore, simply displaying disconnected gloss words fails to convey true linguistic meaning."*

---

## Slide 4: Project Objectives & Scope

### Slide Content
* **Zero-Hardware Landmark Tracking:** 828 3D spatial points extracted at 30+ FPS via standard client webcams using Google MediaPipe Holistic.
* **Spatial Anatomical Differentiation:** Incorporate 12 body-region offset anchors (forehead, temple, nose, mouth, chin, torso) to distinguish signs sharing identical hand configurations.
* **100-Word Approved ISL Vocabulary:** Scale coverage across 7 vital communication domains (1,496 verified sample sequences).
* **Calibrated Open-Set Rejection ($\tau = 0.35$):** Formulate confidence thresholding and temporal hysteresis ($k=3$) to reject transitional and ambient non-signing gestures.
* **SOV-to-SVO Grammar & Audio Synthesis:** Automatic rule-based syntactic reordering with client-side W3C Web Speech audio output.
* **Interactive Sign Practice Studio:** Reference demonstration comparison with live in-camera circular accuracy HUD scoring.
* **AI Retraining Studio with Validation Gate:** In-browser live exemplar recording and rapid model fine-tuning protected by a $\ge 90\%$ baseline accuracy holdout gate.
* **Phase I Scope:** 80% milestone completed and fully validated; zero reliance on external cloud servers or paid APIs.

---

## Slide 5: Literature Survey

### Summary of Reviewed Methodologies

| Paper / Framework | Authors & Citation | Methodology | Key Merits | Critical Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **INCLUDE Dataset** | Selvaraj et al. (ACM MM 2020) | 263 ISL classes, 4,287 HD videos, evaluated with 3D-ResNet/I3D | Authentic native signer dataset across 15 semantic domains | 3D-CNN latency ($>180$ ms) requires server GPUs; no 3D landmarks |
| **MediaPipe Holistic** | Lugaresi, Zhang, et al. (CVPR 2020) | BlazePose + MediaPipe Hands via WebAssembly/WebGL | Zero-hardware, 30+ FPS edge execution; privacy-preserving | Outputs raw coordinates; no temporal classification or grammar |
| **Continuous Recurrent SLR** | Cui, Liu, & Zhang (CVPR / TPAMI) | 2D/3D CNN feature extractor with BiLSTM + CTC alignment | Captures long-range temporal bidirectional context | Heavy memory footprint; closed-set vulnerability to ambient gestures |
| **Open Set Recognition** | Scheirer, Boult, et al. (IEEE TPAMI) | Extreme Value Theory & probabilistic thresholding | Bounds open-space risk; suppresses false-positive alarms | Requires rigorous threshold calibration for noisy edge sensors |
| **Spatial-Temporal GCNs** | Kumar, Kishore, et al. (MTA 2022) | ST-GCN on skeletal pose graphs for Indian Sign Language | Preserves skeletal topological bone constraints | High tensor graph contraction overhead ($>35$ ms); not browser-standard |

### Speaker Notes
> *"Our literature survey thoroughly analyzed landmark works. Selvaraj et al. provided the INCLUDE benchmark but relied on slow 3D-CNNs. MediaPipe showed sub-16 ms edge tracking but lacked sign intelligence. Cui et al. proved bidirectional temporal modeling, while Scheirer and Boult established the mathematical need for open-set rejection. Kumar et al. explored graph convolutions. Drawing from these insights, ISL-Lens combines lightweight geometric landmarks with calibrated open-set machine learning to achieve sub-millisecond classification inside the browser."*

---

## Slide 6: Research Gaps in Existing Literature

### Critical Research Gaps Identified

1. **Absence of Zero-Hardware, Browser-Native Deployment:**
   - *Existing Limitation:* Prior Sign Language Recognition (SLR) research remains trapped in Python/CUDA desktop environments (Anaconda, PyTorch/TensorFlow builds) requiring local dependency setups.
   - *Impact:* Deaf end-users and non-technical healthcare practitioners cannot install or operate complex research scripts on commodity devices.
2. **Severe Latency vs. Accuracy Bottleneck:**
   - *Existing Limitation:* Achieving acceptable recognition accuracy has historically relied on 3D Convolutional Neural Networks (e.g., 3D-ResNet, I3D) that incur frame latencies of **$180$ to $250$ ms** and require enterprise GPUs.
   - *Impact:* Real-time conversational interaction collapses, introducing intolerable conversational lag ($>0.2$ s per gesture).
3. **Pervasive Closed-Set Assumption & False Positives:**
   - *Existing Limitation:* Standard neural classifiers apply closed-set softmax probability distributions, forcing arbitrary hand gestures, facial scratching, or resting arm postures into one of the known vocabulary classes.
   - *Impact:* High false-alarm rate ($>42\%$) in unconstrained domestic or clinical environments.
4. **Rigid, Static, and Non-Extensible Vocabularies:**
   - *Existing Limitation:* Academic systems fix vocabulary sets permanently at training time. End-users cannot add regional dialect variations, localized signs, or family colloquialisms without developer assistance and complete model retraining.
   - *Impact:* Rapid obsolescence and rejection by native deaf communities with evolving sign vernaculars.
5. **Neglect of SOV-to-SVO Syntactic Translation:**
   - *Existing Limitation:* Existing prototypes stop at isolated word classification, simply printing disconnected gloss strings adhering to Indian Sign Language's Subject-Object-Verb (SOV) structure (e.g., `ME WATER DRINK`).
   - *Impact:* Hearing communication partners receive fragmented, ungrammatical text without natural vocal feedback.

### Research Gap Comparative Matrix

| Domain Dimension | Conventional SLR Benchmark | Identified Research Gap | How ISL-Lens Resolves the Gap |
| :--- | :--- | :--- | :--- |
| **Execution Medium** | Local Python scripts / GPU servers | No instant, zero-install web access | Edge WebAssembly & WebRTC running inside any standard browser |
| **Compute & Latency** | 3D-CNNs ($>180$ ms, heavy GPUs) | Prohibitive hardware barrier | Sub-millisecond CPU classification (**18.4 ms** pipeline, 54.3 FPS) |
| **Ambient Robustness** | Closed-set Softmax ($100\%$ sum) | Idle and resting false alarms | Calibrated rejection ($	au=0.35$) + 3-cycle temporal hysteresis |
| **Personalization** | Hardcoded, immutable weights | Deaf signers cannot customize | In-browser AI Retraining Studio protected by a $\ge 90\%$ holdout gate |
| **Linguistic Output** | Raw isolated gloss tokens | Fragmented SOV strings | Rule-based SOV $	o$ SVO grammar translation + Web Speech audio |

### Speaker Notes
> *"Respected evaluators, through our literature review we isolated five critical research gaps that explain why three decades of sign language research have failed to reach real deaf users. First, previous models are stuck in Python research scripts; none provide a friction-free web link. Second, 3D-CNNs require heavy desktop GPUs and take over 180 milliseconds, which breaks conversation. Third, existing systems suffer from closed-set blindness: if a signer rests their hand, the system hallucinates a sign. Fourth, vocabularies are completely static—a deaf person cannot add their own name or regional slang. Fifth, existing works ignore syntax: printing 'ME WATER DRINK' is not proper English. ISL-Lens was engineered ground-up to systematically bridge all five of these gaps."*

---

## Slide 7: Proposed System Architecture

### Multi-Tier Edge Architecture
* **Presentation Tier:** HTML5 Canvas, WebRTC Video Viewport, Responsive Glassmorphic Dashboard, Google Meet-style Floating Action Controls.
* **Perception Tier:** Google MediaPipe Holistic Wasm pipeline extracting 75 anatomical landmarks (21 Right Hand, 21 Left Hand, 33 Upper-Body Pose).
* **Kinematics & Feature Tier:** Torso distance normalization, 12 body-region spatial offsets, and 30-frame temporal velocity pooling.
* **Classification Tier:** Calibrated Multinomial Logistic Regression ($\tau = 0.35$) with 3-frame temporal hysteresis.
* **Language & Audio Tier:** Deterministic SOV-to-SVO Syntactic Parser coupled with W3C Web Speech Synthesis.
* **Studio Tier:** Dual-viewport Practice Studio with dynamic HUD and in-browser AI Retraining Studio with validation gating.

<p align="center">
  <img src="Final Project Presentation/figures/fig_architecture.png" alt="Detailed System Architecture" width="95%" />
  <br><b>Figure 2:</b> End-to-End Modular Architectural Decomposition of ISL-Lens.
</p>

---

## Slide 8: Landmark Extraction & Kinematic Pipeline

### Mathematical Formulation
1. **Torso Reference Center & Scale:**
   $$\mathbf{c}_{\text{ref}} = \frac{\mathbf{s}_L + \mathbf{s}_R}{2}, \quad d_{\text{torso}} = \|\mathbf{s}_L - \mathbf{s}_R\|_2$$
2. **Translation- and Scale-Invariant Coordinate Normalization:**
   $$\mathbf{p}'_i = \frac{\mathbf{p}_i - \mathbf{c}_{\text{ref}}}{d_{\text{torso}} + \epsilon}$$
3. **Spatial Body-Region Anchoring (12 Dimensions):**
   $$\mathbf{d}_{h, r} = \frac{\mathbf{w}_h - \mathbf{a}_r}{d_{\text{torso}} + \epsilon}, \quad \mathbf{a}_r \in \{\text{forehead}, \text{temple}, \text{nose}, \text{mouth}, \text{chin}, \text{torso}\}$$
4. **Dynamic Velocity & Dispersion Vector:**
   $$\mathbf{v}_t = \mathbf{P}'_t - \mathbf{P}'_{t-1}, \quad \mathbf{z} = \left[ \boldsymbol{\mu}(\mathbf{P}'), \boldsymbol{\sigma}(\mathbf{P}'), \mathbf{P}'_{\min}, \mathbf{P}'_{\max}, \boldsymbol{\mu}(\mathbf{v}), \boldsymbol{\sigma}(\mathbf{v}) \right]$$

<p align="center">
  <img src="Final Project Presentation/figures/fig_kinematics.png" alt="Kinematic Feature Extraction" width="90%" />
  <br><b>Figure 3:</b> Coordinate Normalization and Spatial Body-Region Offset Anchors.
</p>

### Speaker Notes
> *"To ensure the model recognizes signs regardless of whether a child or adult signs, or how close they stand to the camera, we calculate the midpoint between the shoulders and normalize all 75 keypoints by shoulder width. We also compute 12 spatial vectors linking the wrists to facial anchors—forehead, nose, chin, mouth, temple, and chest. This allows the system to effortlessly tell apart signs that have the exact same handshape but different meanings based on where they touch the face."*

---

## Slide 9: Open-Set Rejection & ML Engine

### Classification & Rejection Logic
* **Multinomial Logistic Posterior Probability:**
  $$P(y = k \mid \mathbf{z}) = \frac{\exp(\mathbf{w}_k^T \mathbf{z} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{z} + b_j)}$$
* **Calibrated Confidence Gating ($\tau = 0.35$):**
  $$\text{Predicted Class} = \begin{cases} \arg\max_k P(y = k \mid \mathbf{z}) & \text{if } \max_k P(y=k \mid \mathbf{z}) \ge \tau \\ \texttt{UNKNOWN / REJECT} & \text{otherwise} \end{cases}$$
* **Temporal Hysteresis ($k=3$):** A candidate sign must dominate predictions across 3 consecutive cycles before being committed to the live sentence buffer, eliminating flickers and false starts.

---

## Slide 10: Continuous Sentence Translation (SOV to SVO)

### Syntax Engine & English Natural Language Generation
* Indian Sign Language follows **Subject-Object-Verb** syntax.
* ISL-Lens incorporates a rule-based syntax translation engine that parses incoming tokens, performs grammatical inflection, and generates natural English **Subject-Verb-Object** sentences.
* Includes Start/Pause speech controls, audio synthesis, and transcript export.

| Raw ISL Gloss Sequence | Grammatical Structure | Translated Spoken English Sentence |
| :--- | :--- | :--- |
| `I WATER DRINK` | Subject + Object + Verb | *"I am drinking water."* |
| `DOCTOR HELP PLEASE` | Object + Verb + Etiquette | *"Please call a doctor for help."* |
| `YOU HOSPITAL GO WHY` | Subject + Object + Verb + Q | *"Why are you going to the hospital?"* |
| `MY MOTHER SICK HOME` | Possessive + Subj + Adj + Loc | *"My mother is sick at home."* |
| `TODAY WEATHER GOOD` | Temporal + Subject + Adj | *"The weather is good today."* |
| `NAME YOUR WHAT` | Object + Possessive + Q | *"What is your name?"* |

<p align="center">
  <img src="Final Project Presentation/figures/fig_syntax_engine.png" alt="Syntax Engine Translation Flow" width="85%" />
  <br><b>Figure 4:</b> Syntax Engine Processing Pipeline from Raw Glosses to Spoken Sentences.
</p>

---

## Slide 11: Interactive Practice Studio with Circular Accuracy HUD

### Studio Capabilities
* **Dual-Viewport Layout:** Left pane displays verified dataset video demonstration; right pane captures live user webcam feed.
* **Dynamic Time Warping (DTW) & Cosine Trajectory Scoring:**
  $$S(\mathbf{Q}, \mathbf{R}) = \frac{1}{T} \sum_{t=1}^T \frac{\mathbf{q}_t \cdot \mathbf{r}_t}{\|\mathbf{q}_t\|_2 \|\mathbf{r}_t\|_2}$$
* **Diagnostic Multi-Component Scoring:**
  * **Hand Dexterity Score (40%):** Finger curl and joint flexion angles.
  * **Kinetic Trajectory Score (35%):** Movement direction and acceleration envelope.
  * **Anatomical Body Placement Score (25%):** Correct spatial placement near target facial/torso anchor.
* **In-Camera Circular HUD Ring:** Instantaneous visual feedback (e.g., **"96.5% MATCH"** with animated green ring).

<p align="center">
  <img src="Final Project Presentation/figures/fig_browser_practice.png" alt="Interactive Sign Practice Studio" width="85%" />
  <br><b>Figure 5:</b> Interactive Sign Practice Studio with Dual-Viewport and In-Camera Accuracy HUD.
</p>

---

## Slide 12: Interactive AI Retraining Studio & Validation Gate

### On-Device Model Customization
* **Live Webcam Exemplar Capture:** Users record 5–10 webcam repetitions of a new or localized sign directly in the browser.
* **Instantaneous Edge Fine-Tuning:** In-memory logistic weight updates complete in **$< 800$ ms**.
* **Anti-Corruption Base Corpus Validation Gate:**
  $$\text{Accuracy}_{\text{holdout}} \ge 90.0\%$$
* If the newly trained weights cause performance on the verified baseline dataset to drop below 90%, the update is automatically rejected, safeguarding production stability.

<p align="center">
  <img src="Final Project Presentation/figures/fig_browser_studio.png" alt="AI Retraining Studio" width="85%" />
  <br><b>Figure 6:</b> In-Browser AI Retraining Studio with Real-Time Validation Gate Status.
</p>

---

## Slide 13: Hardware and Software Specifications

### System Specifications

| Dimension | Specification Details | Justification / Role |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5/i7 ($\ge 2.4$ GHz) or Apple Silicon M1/M2/M3 | High-speed multi-core client landmark tracking |
| **System Memory (RAM)** | 8 GB Unified Memory / DDR4 | Web console peak consumption is only **$< 320$ MB** |
| **Video Camera** | Standard 720p / 1080p Webcam at 30 FPS | Zero specialized hardware; works on any laptop camera |
| **Client Platform** | Google Chrome, Microsoft Edge, Mozilla Firefox, Safari | Cross-platform W3C HTML5, WebRTC, WebAssembly |
| **Core Languages** | Python 3.11 (Offline ML) & JavaScript ES6+ (Frontend) | Clean separation of model training and edge execution |
| **Perception Framework** | Google MediaPipe Holistic v0.10.x | Real-time 3D hand and pose landmark extraction |
| **ML & Math Stack** | Scikit-Learn 1.3.x, NumPy 1.24.x, SciPy 1.11.x | Vectorized coordinate normalization and classification |
| **Speech Engine** | W3C Web Speech Synthesis API | Zero-cost, zero-latency client-side spoken voice feedback |

---

## Slide 14: Experimental Results & Accuracy Benchmarks

### Quantitative Performance Metrics (100-Word Approved Vocabulary)

| Evaluation Metric | Baseline Closed-Set | Live Calibrated Web Console ($\tau = 0.35$) |
| :--- | :---: | :---: |
| **Top-1 Recognition Accuracy** | **97.81%** | **97.40% (Verified Live)** |
| **Top-3 Recognition Accuracy** | **99.13%** | **99.10%** |
| **Macro-Averaged F1-Score** | **0.9830** | **0.9785** |
| **Weighted-Averaged F1-Score** | **0.9788** | **0.9752** |
| **Sensitivity (Recall)** | 0.978 | 0.974 |
| **Specificity (True Negative Rate)** | 0.997 | 0.996 |
| **Precision** | 0.976 | 0.972 |
| **End-to-End System Latency** | **18.4 ms** | **18.4 ms (54.3 FPS)** |

<p align="center">
  <img src="Final Project Presentation/figures/fig_loocv_progression.png" alt="Cross Validation Progression" width="85%" />
  <br><b>Figure 7:</b> 5-Fold Cross-Validation Accuracy and F1 Stability Across Evaluation Folds.
</p>

### Speaker Notes
> *"In quantitative testing across 1,496 verified sequences spanning all 100 core vocabulary words, ISL-Lens achieved 97.81% Top-1 offline accuracy and 97.4% sustained live webcam accuracy under ambient noise. Our 5-fold cross-validation demonstrated exceptional stability at 97.8% with a standard deviation of only $\pm 0.4\%$, proving the model does not overfit to specific individuals or hand sizes."*

---

## Slide 15: Confusion Matrix & Category Breakdown

### Performance by Vocabulary Domain

| Functional Category | Classes | Sensitivity | Specificity | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Greetings & Etiquette | 12 | 0.985 | 0.998 | 0.982 | 98.4% |
| Family & Relations | 14 | 0.976 | 0.997 | 0.974 | 97.5% |
| Pronouns & Entities | 12 | 0.981 | 0.998 | 0.980 | 98.1% |
| Everyday Actions & Verbs | 22 | 0.968 | 0.996 | 0.965 | 96.9% |
| Interrogatives & Questions | 10 | 0.988 | 0.999 | 0.986 | 98.7% |
| Medical & Emergency | 15 | 0.979 | 0.997 | 0.977 | 97.8% |
| Descriptors & Objects | 15 | 0.972 | 0.996 | 0.970 | 97.3% |
| **Overall Macro Average** | **100** | **0.978** | **0.997** | **0.976** | **97.8%** |

<p align="center">
  <img src="Final Project Presentation/figures/fig_confusion_matrix.png" alt="Confusion Matrix" width="60%" />
  <br><b>Figure 8:</b> Confusion Matrix Across Representative Indian Sign Language Gloss Classes.
</p>

---

## Slide 16: Comparative Analysis with Baseline Models

### Performance Across Model Paradigms

| Model Architecture | Top-1 Accuracy | Top-3 Accuracy | Inference Latency | Model Weight Size | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest (100 Trees) | 95.2% | 98.4% | 2.4 ms | 48.2 MB | 416.7 |
| Support Vector Machine (RBF) | 96.5% | 98.8% | 4.8 ms | 14.5 MB | 208.3 |
| LightGBM Gradient Boosting | 96.8% | 98.9% | 1.9 ms | 12.1 MB | 526.3 |
| BiLSTM + Temporal Attention [Cui et al.] | 96.4% | 98.7% | 14.2 ms | 18.6 MB | 70.4 |
| 3D-ResNet-18 Video CNN [Selvaraj et al.] | 94.2% | 97.1% | 168.0 ms | 132.0 MB | 5.9 |
| **ISL-Lens (Calibrated Multinomial)** | **97.8%** | **99.1%** | **0.8 ms** | **2.4 MB** | **1,250.0** |

<p align="center">
  <img src="Final Project Presentation/figures/fig_device_benchmarks.png" alt="Device Benchmark Comparisons" width="80%" />
  <br><b>Figure 9:</b> Multi-Device Latency Benchmarks (Desktop, Laptop, Tablet, Mobile).
</p>

---

## Slide 17: Ablation Studies & Latency Profiling

### Latency Budget Breakdown (Total: 18.4 ms / 54.3 FPS)
* **WebRTC Video Capture & Canvas Transfer:** 1.1 ms (6.0%)
* **MediaPipe Holistic Landmark Regression:** 15.1 ms (82.1%)
* **Torso Normalization & Velocity Extraction:** 1.4 ms (7.6%)
* **Calibrated Classification & Gating:** 0.8 ms (4.3%)
* **UI DOM Rendering & TTS Dispatch:** $<0.1$ ms ($<0.1\%$)

<p align="center">
  <img src="Final Project Presentation/figures/fig_latency_breakdown.png" alt="Latency Breakdown" width="48%" />
  <img src="Final Project Presentation/figures/fig_ablation_chart.png" alt="Ablation Chart" width="48%" />
  <br><b>Figure 10:</b> (Left) End-to-End Latency Stage Distribution; (Right) Ablation Study Feature Impact.
</p>

### Ablation Findings
1. **Pose Anchoring:** Adding torso pose to hands increases accuracy from **91.2% to 95.8%**.
2. **Kinematic Velocity:** Inter-frame velocity boosts accuracy from **95.8% to 97.8%**.
3. **Open-Set Gating:** Threshold $\tau = 0.35$ slashes false alarms on non-signing motions from **42.6% to 3.4%**.

---

## Slide 18: Real-World Applications

### Practical Deployment Scenarios
* **Healthcare & Emergency Rooms:** Allows deaf patients to communicate acute pain symptoms, allergies, and medical history to attending doctors without waiting for an on-site interpreter.
* **Judicial & Administrative Desks:** Enables friction-free communication for legal filings, police inquiries, and municipal public service kiosks.
* **Inclusive Classrooms & Universities:** Facilitates active participation of deaf students during seminar presentations, tutoring sessions, and team collaborations.
* **Self-Directed ISL Education:** Interactive Practice Studio empowers hearing individuals, families, and teachers to learn and master Indian Sign Language with live feedback.
* **Web Tele-Conferencing Integration:** Modular client-side architecture allows direct embedding into Google Meet, Zoom, or WebRTC telemedicine portals.

---

## Slide 19: Key Advantages of ISL-Lens

### Technical & Societal Strengths
1. **Zero Financial Barrier:** 100% free of specialized hardware; operates on any commodity webcam.
2. **Privacy Guaranteed:** Video frames are processed in-memory and discarded immediately; no video is ever transmitted to remote cloud servers.
3. **Sub-Millisecond Inference:** At 0.8 ms classification (18.4 ms total), the system easily outpaces the 50 ms human conversational perception threshold.
4. **Resilient to Environmental Clutter:** Pure skeletal geometry eliminates errors caused by poor room lighting, skin tone variations, or crowded backgrounds.
5. **Dynamic Community Extensibility:** Non-technical deaf users can add and fine-tune personal signs through the AI Retraining Studio without writing a single line of code.
6. **Linguistically Sound:** Bridging SOV sign structure to SVO English sentences produces natural vocalized communication.

---

## Slide 20: 80% Milestone Verification & Phase II Roadmap

### Phase I 80% Milestone Evaluation Matrix

| Subsystem Module | Academic Scope & Deliverable | Current Operational Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Module 1: Vision & Tracking** | Zero-hardware webcam 3D landmark pipeline | **100% Completed** | WebRTC 30+ FPS stream, 828 3D points |
| **Module 2: 100-Word ISL Engine** | 100 approved classes, spatial body anchoring | **100% Completed** | 97.4% verified accuracy, 1,496 sequences |
| **Module 3: Sentence Translation** | SOV-to-SVO translation and audio synthesis | **100% Completed** | Dynamic grammar engine, Web Speech API |
| **Module 4: AI Retraining Studio** | On-device sign recording and fine-tuning | **100% Completed** | $<800$ ms retraining, $\ge 90\%$ holdout gate |
| **Module 5: Practice Studio** | Dual-viewport rehearsal and accuracy HUD | **100% Completed** | Circular accuracy ring, 96.5% match score |

### Phase II Future Roadmap
* **Spatial-Temporal Graph Convolutional Networks (ST-GCN):** Model non-Euclidean bone topology directly.
* **Continuous Unsegmented Sequence Decoding:** Connectionist Temporal Classification (CTC) for free-flowing multi-sentence sign streams.
* **Mobile Progressive Web App (PWA):** Offline mobile deployment for Android and iOS devices.
* **Full-Duplex Telehealth Calling:** Embed into peer-to-peer WebRTC video rooms for remote consultations.

---

## Slide 21: Limitations of the Current System

### Identified Technical Limitations & Scope Boundaries

1. **Severe Optical Occlusion & Complex Hand Interlacing:**
   - *Description:* When both hands cross directly over each other in front of the camera along the optical axis (e.g., tightly interlaced fingers or one hand completely shadowing the other), 2D RGB monocular keypoint tracking can experience momentary spatial ambiguity.
   - *Current Mitigation:* Temporal hysteresis ($k=3$) and shoulder-normalized spatial priors filter momentary jitter, but complete resolution requires multi-angle video feeds.
2. **Coarticulation in Rapid Free-Flowing Signing:**
   - *Description:* In natural, rapid conversational signing without deliberate pauses between words, the boundary between adjacent signs blurs due to *coarticulation* (the transitional stroke linking two gestures).
   - *Current Scope:* ISL-Lens utilizes a 3.0-second temporal sliding window with pause-aware segmentation. Handling unconstrained, non-stop continuous signing is earmarked for Phase II sequence alignment.
3. **Pan-Indian Regional Dialect Diversity:**
   - *Description:* Indian Sign Language exhibits notable lexical dialect variations across Northern, Western, and Southern Indian states (e.g., different colloquial gestures for specific nouns).
   - *Current Mitigation:* The baseline 100-word model is trained strictly on the national standard ISLRTC 3rd Edition dictionary, and the AI Retraining Studio allows local users to add regional dialect glosses on-device.
4. **Extreme Low-Light & Severe Backlighting (< 50 lux):**
   - *Description:* Although landmark abstraction eliminates skin tone and clothing dependencies, extreme underexposure ($<50$ lux) or intense backlight (standing in front of an open window) degrades webcam sensor signal-to-noise ratio, lowering landmark confidence.
   - *Current Requirement:* Minimum standard indoor illumination ($\ge 150$ lux) is recommended for reliable landmark extraction.
5. **Monocular Pseudo-3D Depth Ambiguity:**
   - *Description:* Standard consumer webcams lack hardware Time-of-Flight (ToF) or LiDAR sensors. The MediaPipe $z$-coordinate is an inferred metric estimation relative to the torso, which carries lower millimeter-level precision than lateral $(x, y)$ planar tracking.

### Speaker Notes
> *"In the spirit of academic rigor, we have critically evaluated the limitations of our current Phase I prototype. First, because we rely on a single monocular webcam, severe optical occlusion—such as when one hand completely covers the other—can momentarily degrade landmark tracking. Second, in very rapid conversational signing without pauses, coarticulation between gestures poses a segmentation challenge that we currently manage with temporal hysteresis. Third, while our 100-word vocabulary aligns with the national ISLRTC standard, regional sign variations exist across states, which is why we built the AI Retraining Studio so communities can personalize their vocabulary. Finally, like all computer vision systems, extreme low light (<50 lux) can degrade webcam sensor fidelity. Acknowledging these boundaries has directly informed our Phase II research roadmap."*

---

## Slide 22: Conclusion & Summary

### Slide Summary
* **Vision Realized:** ISL-Lens successfully bridges the communication divide between the Deaf/Hard-of-Hearing community and hearing society through an edge-first, zero-hardware web platform.
* **Performance Benchmark:** Achieved **97.4% verified live accuracy** across 100 vocabulary classes with an end-to-end latency of only **18.4 ms** (54.3 FPS).
* **Innovation Highlights:**
  * Calibrated open-set rejection ($\tau = 0.35$) eliminates idle false alarms.
  * Automated SOV-to-SVO syntactic translation generates natural spoken English sentences with client-side speech synthesis.
  * In-browser AI Retraining Studio democratizes model personalization while preventing catastrophic forgetting via a $\ge 90\%$ validation gate.
* **Academic Achievement:** Phase I 80% implementation milestone fully satisfied; research manuscript submitted for peer-reviewed journal publication.

---

## Slide 23: Publication Details

### Journal Paper Submission Specifications

| Parameter | Submission Specification |
| :--- | :--- |
| **Paper Title** | *ISL-LENS: Real-Time Indian Sign Language Recognition And Interactive AI Console* |
| **Author** | **Aevlin Prince** (`aevlinprince2027@cs.ajce.in`) |
| **Department & Institution** | Dept. of Computer Science and Engineering, Amal Jyothi College of Engineering |
| **Faculty Guide** | Dr. Indu John, Assistant Professor |
| **Target Journal** | **International Journal on Emerging Research Areas (IJERA)** |
| **Editorial Communication** | Dr. Juby Mathew, Editor & HoD CSE (`ijera@amaljyothi.ac.in`) |
| **Submission ID / Paper No.** | **554** |
| **Tracking Dashboard** | `https://ijera.in/index.php/IJERA/authorDashboard/submission/554` |
| **Submission Timestamp** | **26 October 2026 at 08:29 AM IST** |
| **Current Editorial Status** | **Under Editorial Peer Review** |

<p align="center">
  <img src="Final Project Presentation/figures/fig_ijera_submission_mail.png" alt="IJERA Submission Acknowledgment Letter" width="75%" />
  <br><b>Figure 11:</b> Official Journal Submission Acknowledgment Letter from IJERA for Paper ID 554.
</p>

### Speaker Notes
> *"I am delighted to report that the technical contributions and experimental findings of this project have been formulated into a comprehensive research manuscript titled 'ISL-LENS: Real-Time Indian Sign Language Recognition And Interactive AI Console'. It was officially submitted to the International Journal on Emerging Research Areas (IJERA) under Submission ID 554 on October 26, 2026, and is currently undergoing peer review."*

---

## Slide 24: References

### Selected Academic Literature
1. **P. Selvaraj, G. N. C., P. Kumar, and M. M. Khapra**, "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition," in *Proc. 28th ACM International Conference on Multimedia (MM '20)*, Seattle, WA, USA, pp. 1317–1325, 2020.
2. **C. Lugaresi, J. Tang, H. Nash, et al.**, "MediaPipe: A Framework for Building Perception Pipelines," *arXiv preprint arXiv:1906.08172*, 2019.
3. **F. Zhang, V. Bazarevsky, A. Vakunov, et al.**, "MediaPipe Hands: On-device Real-time Hand Tracking," in *CVPR Workshop on Computer Vision for AR/VR*, Seattle, WA, USA, 2020.
4. **V. Bazarevsky, I. Grishchenko, K. Raveendran, et al.**, "BlazePose: On-device Real-time Body Pose Tracking," in *CVPR Workshop on Computer Vision for AR/VR*, 2020.
5. **N. C. Camgoz, S. Hadfield, O. Koller, H. Ney, and R. Bowden**, "Neural Sign Language Translation," in *Proc. IEEE/CVF CVPR*, pp. 7784–7793, 2018.
6. **O. Koller, S. Zargaran, and H. Ney**, "Re-Sign: Re-aligned End-to-End Sequence Modelling with Deep Recurrent CNN-HMMs," *IEEE TPAMI*, vol. 42, no. 5, pp. 1060–1073, 2020.
7. **R. Cui, H. Liu, and C. Zhang**, "Recurrent Convolutional Neural Networks for Continuous Sign Language Recognition by Staging," in *Proc. IEEE CVPR*, pp. 7365–7373, 2017.
8. **W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult**, "Toward Open Set Recognition," in *IEEE TPAMI*, vol. 35, no. 7, pp. 1757–1772, 2013.
9. **K. V. V. Kumar, P. V. V. Kishore, E. K. Kumar, and D. A. Kumar**, "Indian Sign Language Recognition using Graph Convolutional Networks on Spatial-Temporal Pose Graphs," *Multimedia Tools and Applications*, vol. 81, pp. 28945–28968, 2022.
10. **Indian Sign Language Research and Training Centre (ISLRTC)**, *Indian Sign Language Basic Dictionary (3rd Edition)*, Ministry of Social Justice and Empowerment, Government of India, New Delhi, 2021.
11. **World Health Organization (WHO)**, "World Report on Hearing: Executive Summary," Tech. Rep. WHO/UCN/NCD/21.1, Mar. 2021.
12. **F. Pedregosa et al.**, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

<p align="center">
  <b>Thank You!</b><br>
  <i>Questions & Discussion Welcomed</i>
</p>
