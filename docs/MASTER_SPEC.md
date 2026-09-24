# ISL-Lens Master Specification

## 1. Project Purpose

ISL-Lens is a real-time Indian Sign Language recognition and communication web application. Its primary goal is to recognize supported ISL signs accurately from webcam video and support two-way communication between signing and speaking users.

The project must be technically defensible as a final-year B.Tech engineering project. It must prioritize measured recognition quality, robust rejection of unsupported gestures, reproducibility, and honest reporting over demo-only feature count.

The central recognition objective is:

- recognize known supported signs as accurately as possible;
- return `UNKNOWN` for unsupported, incomplete, ambiguous, or low-quality gestures;
- measure dataset performance and real-world webcam performance separately;
- avoid reporting any metric that has not actually been measured.

The system must not claim or fake 100% accuracy. The target is to design and evaluate the strongest practical system possible for the supported vocabulary.

## 2. Roadmap Requirements

The original roadmap establishes the following intended capabilities:

- webcam-based ISL recognition;
- MediaPipe landmark extraction;
- deep-learning sign recognition;
- originally proposed BiLSTM + Attention architecture;
- Flask-based website;
- browser webcam input;
- skeleton visualization;
- raw recognized sign output;
- text-to-speech;
- running transcript;
- vocabulary panel;
- speech-to-text reverse communication;
- practice mode;
- support for 100+ signs over time;
- PDF export of conversations;
- future sentence generation;
- future WebRTC support;
- future LLM grammar correction;
- future federated learning.

The original roadmap also includes target metrics:

- greater than 90% recognition accuracy;
- greater than 97% Top-3 accuracy;
- less than 8% false positive rate;
- less than 60 ms per frame inference;
- less than 300 ms sign-to-speech latency;
- greater than 75 SUS usability score.

These are project targets, not achieved results. They must not be copied into reports as measured outcomes unless experiments produce them.

## 3. Requirements We Are Changing

The rebuilt project changes several assumptions from the original roadmap:

- BiLSTM + Attention is a candidate model, not the guaranteed final model.
- The feature representation is not fixed to `30 frames x 1662 values`.
- TensorFlow.js is not assumed as the deployment target until model and latency evidence support it.
- Dataset audit comes before model implementation.
- Isolated sign recognition and sentence generation are treated as separate machine learning problems.
- Open-set rejection is mandatory, not optional.
- Real-world webcam evaluation is required in addition to offline dataset evaluation.
- Vocabulary must be configuration/data driven and must not be hardcoded across multiple files.
- Reported accuracy must distinguish standard benchmark splits from signer-independent splits where possible.

## 4. System Architecture

The recognition pipeline is:

```text
Camera
-> frame capture
-> landmark extraction
-> canonical preprocessing
-> temporal sequence construction
-> spatial/temporal model
-> class probabilities and/or embedding
-> open-set rejection
-> temporal decision engine
-> committed sign
-> sign sequence
-> language/sentence layer
-> natural English
-> text-to-speech
```

The speech-to-text communication path is separate:

```text
Microphone
-> speech recognition
-> text
-> conversation interface
```

The architecture must keep these concerns separate:

- landmark extraction;
- canonical preprocessing;
- model inference;
- open-set rejection;
- temporal decision logic;
- language/sentence generation;
- web application presentation;
- conversation export.

Model logic must not be embedded directly in HTML. Flask may serve the application, APIs, configuration, export functionality, and later sentence services, but recognition logic must live in dedicated source modules.

## 5. Dataset Strategy

Dataset-first development is mandatory. Before training any model, the project must audit candidate datasets and create a machine-readable manifest that becomes the source of truth for training and evaluation.

The dataset audit must record:

- source and download method;
- license and usage restrictions;
- vocabulary/classes;
- number of samples/videos;
- number of signers;
- signer metadata when available;
- sequence lengths;
- class imbalance;
- isolated versus continuous signing;
- annotation format;
- gloss labels;
- English translations if available;
- train/validation/test split availability;
- signer-independent split availability;
- duplicate or near-duplicate samples;
- corrupted or unusable samples;
- known limitations.

Data leakage prevention is a core requirement. The same signer, recording, source video, near-duplicate frames, or augmented copies must not be allowed to leak across train/validation/test splits in ways that inflate accuracy.

Evaluation should report both:

- standard benchmark result;
- signer-independent result when metadata supports it.

If these differ, the difference must be explicitly discussed.

## 6. Feature Strategy

The project must investigate feature representations instead of blindly copying a fixed dimensionality. Candidate representations include:

- hand landmarks only;
- both hands;
- hands plus pose;
- hands plus pose plus selected face landmarks;
- normalized coordinates;
- relative joint coordinates;
- bone vectors;
- velocities;
- accelerations;
- visibility or presence masks.

The selected feature definition must be justified using validation and signer-independent results where possible. Extra landmarks must be included only if they measurably improve recognition or rejection behavior.

The canonical preprocessing specification must define:

- coordinate normalization;
- reference point;
- scale normalization;
- rotation handling if used;
- handedness handling;
- missing landmark handling;
- temporal resampling;
- padding and masks;
- velocity and acceleration calculation if used;
- feature ordering;
- dtype;
- tensor dimensions.

Training and live inference must use mathematically equivalent preprocessing. Unit tests must verify preprocessing consistency.

## 7. Model Strategy

The original BiLSTM + Attention design is one candidate in a controlled model comparison. Candidate model families include:

- SVM or other simple baselines where technically meaningful;
- CNN or temporal CNN baselines;
- LSTM/BiLSTM baselines;
- BiLSTM + Attention;
- Temporal Transformer;
- Spatial-Temporal Graph Convolutional Network;
- ST-GCN + Temporal Transformer;
- another architecture if dataset evidence justifies it.

Every candidate model must use the same splits, preprocessing, and evaluation protocol. The final model must be selected from experimental evidence, not from perceived sophistication.

Benchmark records must include:

- experiment ID;
- dataset version;
- split ID;
- preprocessing version;
- feature definition;
- random seed;
- model configuration;
- hyperparameters;
- training configuration;
- software versions;
- hardware;
- validation metrics;
- test metrics;
- latency;
- model size;
- memory usage where available.

Old experiments must not be overwritten.

## 8. Open-Set Strategy

Open-set rejection is mandatory. The classifier must not be forced to map every input to a known sign.

The system must return `UNKNOWN` for cases such as:

- unsupported ISL signs;
- random hand motion;
- incomplete signs;
- transition movement between signs;
- resting hands;
- non-sign movement;
- poor landmark detection;
- ambiguous gestures;
- visually similar gestures with insufficient confidence.

Thresholds must not be arbitrary. Rejection thresholds must be learned or selected using validation data and must not be tuned on the final test set.

Candidate rejection mechanisms include:

- calibrated confidence;
- top-1/top-2 margin;
- energy-based rejection;
- embedding distance;
- class centroid distance;
- Mahalanobis distance;
- ensemble disagreement;
- other justified methods.

Open-set evaluation must report:

- false acceptance rate;
- false rejection rate;
- known-sign accuracy;
- unknown rejection accuracy;
- AUROC/AUPR where appropriate;
- confusion between unknown and known classes.

If a gesture is not in the supported vocabulary, it must not be mapped to the closest known sign. To support a new sign, the project must add legitimate data, update the vocabulary, retrain, and reevaluate.

## 9. Temporal Decision Strategy

The model may produce predictions over frames or windows, but the application must not commit every prediction directly to the transcript.

The temporal decision layer must handle:

- sign onset;
- sign completion;
- prediction stability;
- confidence and margin;
- unknown rejection;
- cooldown;
- duplicate suppression;
- pauses;
- transitions;
- incomplete gestures.

Repeated stable predictions such as `HELLO, HELLO, HELLO, HELLO` should commit one `HELLO`, not four. Unstable sequences such as `HELLO, UNKNOWN, GO, UNKNOWN` should not automatically become `HELLO GO` unless the decision engine determines that valid completed signs occurred.

The decision engine must be tested separately from the recognition model.

## 10. Deployment Strategy

The original roadmap suggests TensorFlow.js, but deployment runtime is not fixed yet. Candidate deployment options include:

- TensorFlow.js;
- ONNX Runtime Web;
- backend inference;
- another browser-compatible runtime.

The final deployment choice must consider:

- accuracy preservation after export/conversion;
- latency;
- browser compatibility;
- model size;
- maintainability;
- privacy;
- offline capability;
- hardware support.

The frontend should be designed so the inference runtime can change without rewriting the user interface.

## 11. Sentence-Generation Strategy

Isolated sign recognition and sentence generation are separate problems.

Stage 1:

```text
video -> isolated sign label
```

Stage 2:

```text
sign/gloss sequence -> natural English sentence
```

Before implementing sentence generation, the project must audit whether available data provides a valid supervised pairing such as video or pose data plus gloss sequence plus English sentence. If only isolated labels are available, the project cannot honestly train or evaluate a full continuous sign-to-English translation model.

Sentence generation may later use recognized sign sequences as input, but it must have separate data, evaluation, and limitations. No sentence-generation accuracy should be claimed without ground-truth sentence data.

## 12. Frontend Architecture

The eventual application interface should contain:

- webcam view;
- skeleton overlay;
- recognition status;
- current sign;
- confidence/status indicator;
- explicit `UNKNOWN` state;
- running transcript;
- vocabulary panel;
- TTS controls;
- speech-to-text display;
- practice mode;
- sentence output;
- PDF export.

The interface must clearly distinguish:

- raw recognized signs;
- generated natural-language output;
- speech-to-text messages from the hearing user.

The skeleton overlay is functional, not decorative. It must communicate tracking status, hand visibility, landmark detection quality, and whether the signer is inside the usable camera region.

Practice mode must define any displayed score honestly. It must not show fake precision unless the score has a defensible calculation.

PDF export should include:

- timestamp;
- recognized signs;
- generated sentences;
- speech-to-text messages;
- conversation direction.

Internal model confidence should be included only when useful to the user.

## 13. Testing Strategy

Testing must cover the core correctness risks:

- dataset manifest validation;
- split leakage checks;
- preprocessing equivalence between training and live inference;
- feature shape and dtype consistency;
- missing landmark handling;
- temporal resampling and padding;
- model evaluation scripts;
- open-set threshold selection;
- temporal decision logic;
- duplicate suppression;
- unknown handling;
- API boundaries;
- frontend status states once the web app exists;
- PDF export content once implemented.

Preprocessing tests are especially important because training and live inference must use the same canonical transformation.

## 14. Evaluation Metrics

Recognition model evaluation must include:

- Top-1 accuracy;
- Top-3 accuracy;
- macro F1;
- weighted F1;
- per-class precision;
- per-class recall;
- confusion matrix;
- difficult-class analysis;
- visually similar class analysis;
- short-sequence performance;
- long-sequence performance;
- parameter count;
- model size;
- inference latency;
- CPU latency;
- Apple Silicon/MPS latency if available;
- memory usage where available.

Open-set evaluation must include:

- false acceptance rate;
- false rejection rate;
- known-sign accuracy;
- unknown rejection accuracy;
- AUROC/AUPR where appropriate.

Real-world webcam evaluation must separately test:

- lighting variation;
- background variation;
- camera distance;
- hand position variation;
- clothing variation;
- multiple users/signers;
- left/right variations where applicable;
- slow signing;
- fast signing;
- partial signing;
- transitions;
- pauses;
- unsupported gestures;
- visually similar signs.

Final reporting must distinguish dataset performance from real-world webcam performance.

## 15. Project Structure

The planned structure is:

```text
isl-lens/
  data/
    raw/
    processed/
    manifests/
    splits/
  src/
    preprocessing/
    landmarks/
    models/
    training/
    evaluation/
    inference/
    rejection/
    decision/
  app/
    backend/
    templates/
    static/
  sentence/
    data/
    models/
    inference/
    evaluation/
  tests/
  configs/
  experiments/
  reports/
  scripts/
  docs/
```

This structure is intentionally staged. Gate 0 creates documentation only. Directories and code should be added when their gate begins, not as empty placeholders.

## 16. Development Gates

Development must proceed through gates:

- Gate 0: requirements and specification;
- Gate 1: dataset audit;
- Gate 2: canonical preprocessing;
- Gate 3: baseline recognition model;
- Gate 4: model bakeoff;
- Gate 5: open-set rejection;
- Gate 6: real webcam inference;
- Gate 7: web application;
- Gate 8: sentence generation;
- Gate 9: two-way communication;
- Gate 10: practice mode and PDF export;
- Gate 11: final evaluation.

The project must not jump ahead. Each gate should define artifacts, tests, and completion criteria before moving on.

## 17. Risks

Key risks include:

- insufficient public ISL data for 100+ signs;
- missing signer metadata preventing true signer-independent evaluation;
- data leakage causing inflated accuracy;
- visually similar signs requiring more data than available;
- landmark detection failures under real webcam conditions;
- model overfitting to dataset background or signer identity;
- unsupported gestures being falsely accepted as known signs;
- arbitrary thresholds producing misleading rejection behavior;
- browser deployment reducing model accuracy or increasing latency;
- sentence generation being impossible to evaluate without paired sentence data;
- user interface implying more certainty than the system actually has.

Risk mitigation must emphasize dataset audit, leakage controls, validation-driven thresholds, real-world testing, and clear communication of limitations.

## 18. Definition of Done

The recognition subsystem is not done until:

- dataset audit is complete;
- splits are verified;
- preprocessing is specified and tested;
- at least one baseline is trained;
- multiple justified temporal architectures are compared;
- signer-independent evaluation is performed where possible;
- open-set rejection is implemented;
- rejection is evaluated;
- live webcam inference works;
- temporal duplicate suppression works;
- unknown gestures do not get randomly mapped to known classes;
- real-world webcam testing is completed;
- measured metrics are recorded.

The full application is not done until:

- recognition works;
- skeleton overlay works;
- transcript works;
- TTS works;
- vocabulary is data driven;
- STT works;
- sentence layer works to the extent supported by data;
- practice mode works;
- PDF export works;
- end-to-end testing works.

Final principle:

```text
Accuracy > feature count.
Correct rejection > confident wrong prediction.
Real-world signer-independent performance > impressive training accuracy.
Measured evidence > assumptions.
Clean architecture > quick hacks.
```
