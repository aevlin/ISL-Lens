# ISL-Lens — Real-Time Indian Sign Language Console

An advanced, real-time Indian Sign Language (ISL) gesture recognition, sentence interpreter, and interactive AI model retraining studio built with MediaPipe spatial landmark tracking and scikit-learn machine learning.

## Core Capabilities & Performance Enforcements

- **Held-out sign recognition estimate**: The active model scored 93.8% Top-1 and 97.7% Top-3 on 129 held-out corpus clips. This small same-corpus result is not a guarantee for new signers, continuous signing, or every item in the 100-word vocabulary. The model currently has 48 trained labels.
- **Pause & Translate Sentence Generation Engine**: Click to start recording, perform continuous ISL gesture sequences, click **Pause & Translate Sentence** to load, analyze, and speak out the translated English sentence.
- **Interactive ISL Sign Tutor & Video Presentation**: Matching dataset clips and guide details are shown when verified project assets exist. The app does not substitute a different sign's clip when one is unavailable.
- **AI Teaching & Retraining Studio**: Captures landmark vectors and accepts a model update only when the candidate retains at least 90% Top-1 accuracy on the held-out corpus split.
- **Live prediction score**: The camera view shows the model's score for its current prediction. This score is not the same as measured accuracy.

## Run Locally

```bash
python3 scripts/run_local.py
```

Open in browser:
```text
http://127.0.0.1:8000
```
