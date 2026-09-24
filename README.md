# ISL-Lens

ISL-Lens is a real-time Indian Sign Language recognition and communication web application.

Current status: local working product shell. The app deliberately returns `UNKNOWN` until an audited dataset and trained model are configured.

## Run Locally

```bash
python3 scripts/run_local.py
```

Open:

```text
http://127.0.0.1:8000
```

## What Works Now

- webcam preview;
- browser hand tracking overlay when MediaPipe browser bundles are reachable;
- explicit `UNKNOWN` recognition policy;
- tracking status and hand count;
- running transcript;
- browser text-to-speech;
- browser speech-to-text where supported;
- candidate vocabulary panel;
- practice target flow without fake scoring;
- JSON conversation export.

## What Is Intentionally Not Claimed Yet

- no sign classification accuracy;
- no trained model;
- no sentence-generation accuracy;
- no 100% recognition claim;
- no supported vocabulary until backed by audited training data.
