# ISL-Lens Architectural Decisions

This document records major project decisions and the reason for each. It should be updated whenever a design choice affects accuracy, evaluation validity, deployment, reproducibility, or user-facing behavior.

## Decision 001: Treat BiLSTM + Attention As a Candidate, Not the Final Model

Status: accepted

The original roadmap proposed BiLSTM + Attention. In the rebuilt project, this architecture will be evaluated as one candidate among baselines, temporal transformers, graph-based models, and hybrid approaches.

Reason:

- model choice must be based on measured performance;
- more complex architectures are not automatically better;
- the best model depends on dataset size, signer diversity, feature representation, latency, and deployment constraints;
- final-year reporting must defend the selected model with experimental evidence.

## Decision 002: Make `UNKNOWN` a First-Class Output

Status: accepted

The recognition system must be allowed to reject inputs instead of always selecting a known sign.

Reason:

- supported vocabulary will always be limited;
- unsupported ISL signs must not be mapped to the closest known label;
- ambiguous, incomplete, or low-quality gestures should be rejected rather than guessed;
- a wrong confident sign can harm communication more than an honest `UNKNOWN`;
- open-set behavior is central to real-world accuracy.

## Decision 003: Perform Dataset Audit Before Training

Status: accepted

No model training should begin until candidate datasets are audited and represented in a machine-readable manifest.

Reason:

- dataset quality determines the ceiling of recognition accuracy;
- license and usage constraints must be known early;
- signer metadata is required for signer-independent evaluation;
- class imbalance, duplicates, corrupted samples, and annotation quality affect model validity;
- hardcoded vocabulary or undocumented splits would make experiments difficult to reproduce.

## Decision 004: Prioritize Signer-Independent Evaluation

Status: accepted

Where metadata permits, the project must evaluate on signers not seen during training.

Reason:

- random splits can overestimate accuracy by letting the model learn signer-specific style, camera setup, or background;
- real users will differ from training signers;
- signer-independent performance is more defensible than training/test performance that may contain identity leakage;
- any gap between benchmark and signer-independent results is important evidence for future data collection.

## Decision 005: Keep Isolated Recognition Separate From Sentence Generation

Status: accepted

The project separates isolated sign recognition from continuous sign sequence translation and natural English generation.

Reason:

- recognizing `HELLO` from a video is not the same problem as generating a grammatical English sentence from a signed utterance;
- sentence generation requires suitable paired data, such as gloss sequences and English translations;
- isolated-label datasets cannot directly prove sentence translation quality;
- separate modules make limitations clear and avoid overstating the system.

## Decision 006: Use One Canonical Preprocessing Specification

Status: accepted

Training and live inference must use the same mathematically defined preprocessing pipeline.

Reason:

- mismatched preprocessing can silently destroy live accuracy even when dataset metrics are high;
- normalization, handedness, missing landmarks, masks, sequence length, and feature ordering must remain consistent;
- unit tests can verify stable tensor shapes and transformations;
- a shared preprocessing contract makes experiments reproducible.

## Decision 007: Do Not Fix the Feature Representation Prematurely

Status: accepted

The rebuilt system will not assume the original `30 frames x 1662 values` representation.

Reason:

- thousands of available MediaPipe coordinates may include irrelevant or noisy information;
- ISL recognition may benefit more from normalized hands, pose, velocities, masks, or selected face/body features than from raw full landmarks;
- feature choices must be validated against recognition accuracy, signer-independent performance, and latency;
- smaller well-designed features may generalize better than large noisy tensors.

## Decision 008: Separate Temporal Model Predictions From Transcript Commits

Status: accepted

The application must use a temporal decision engine between model predictions and committed transcript signs.

Reason:

- live models produce repeated, noisy, or unstable window predictions;
- committing every prediction would duplicate signs and create false transcripts;
- onset, completion, confidence, stability, cooldown, transitions, and pauses need explicit logic;
- temporal logic can reduce false positives without changing the underlying classifier.

## Decision 009: Do Not Treat Benchmark Accuracy Alone as Success

Status: accepted

Offline dataset metrics are necessary but not sufficient.

Reason:

- real webcam input may differ from dataset videos in lighting, distance, background, speed, signer identity, and landmark reliability;
- benchmark splits may hide data leakage or signer overlap;
- final evaluation must distinguish dataset performance from real-world webcam performance;
- usability depends on latency, rejection behavior, duplicate suppression, and user feedback as well as classification accuracy.

## Decision 010: Delay Deployment Runtime Selection

Status: accepted

TensorFlow.js, ONNX Runtime Web, and backend inference remain candidate deployment options until model selection and latency testing provide evidence.

Reason:

- conversion may affect accuracy or supported operations;
- browser latency and model size may differ by runtime;
- backend inference may simplify deployment but reduce privacy and offline capability;
- frontend architecture should not be tightly coupled to one model runtime before evidence supports it.

## Decision 011: Keep Vocabulary Data Driven

Status: accepted

The supported vocabulary must come from a single configuration or manifest source instead of being duplicated across Python, JavaScript, HTML, and reports.

Reason:

- the project must scale from small vocabularies to 100+ signs;
- duplicated labels create mismatch risks during training, inference, UI display, and evaluation;
- open-set behavior depends on knowing exactly which signs are supported;
- reproducibility requires each experiment to record the vocabulary version.

## Decision 012: Gate Development to Avoid Premature Implementation

Status: accepted

The project will proceed through development gates, beginning with Gate 0 requirements/specification and Gate 1 dataset audit.

Reason:

- accuracy depends on data and evaluation design before code volume;
- premature frontend or model work can hide foundational problems;
- gates make the project easier to defend, test, and report;
- the current first task is documentation only and must stop after creating the master spec and decision log.

## Decision 013: Use INCLUDE as the First Primary Dataset Candidate

Status: accepted

INCLUDE is selected as the first primary dataset candidate for isolated ISL recognition, subject to local verification of license terms, file integrity, labels, metadata, and split quality.

Reason:

- it directly targets isolated Indian Sign Language recognition;
- its reported 263 classes and 4,287 videos are more appropriate for the project than very small historical datasets;
- the INCLUDE-50 subset can support faster early preprocessing and baseline experiments;
- adopting it as a candidate still avoids assuming that its splits, metadata, or license are sufficient before audit.

## Decision 014: Do Not Use Sentence or Fingerspelling Datasets as the Primary Recognition Dataset

Status: accepted

ISL-CSLTR and ISL-Fingerspelling are valuable future datasets, but they will not drive the first isolated word-sign recognizer.

Reason:

- sentence-level translation and fingerspelling are separate tasks from isolated word-sign recognition;
- using them too early would blur evaluation targets;
- they may become useful in later gates for sentence generation or fingerspelling-specific features;
- the first recognition pipeline needs a clean isolated-sign benchmark before scope expands.

## Decision 015: Treat Third-Party Aggregates and Re-encodes as Unverified Until Provenance Is Checked

Status: accepted

Aggregated or re-encoded datasets may be useful, but they must not be adopted until their upstream licenses, label mappings, duplicate status, and source metadata are checked.

Reason:

- a derived dataset can mix multiple licenses and label conventions;
- re-encoding may change media quality and complicate citation or redistribution;
- label aliases can introduce semantic noise;
- provenance matters for a final-year project that must be defensible.
