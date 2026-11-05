"""Compatibility command for the validated landmark model trainer.

The former implementation evaluated on examples it had already fitted and
could overwrite the production model with a misleading score. Keep this legacy
entry point pointed at the single held-out trainer instead.
"""

from train_calibrated_model import main


if __name__ == "__main__":
    main()
