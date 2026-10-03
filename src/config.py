"""Settings for every experiment.

Sized for a Kaggle T4. IMG_SIZE 128 instead of 224 and 300 images per class
instead of the full set keep a 7-model, 3-seed sweep inside the session limit.
Both cost some accuracy in absolute terms, but every arm pays the same cost,
so the comparison between arms is still fair.
"""

IMG_SIZE = 128
BATCH_SIZE = 32
MAX_PER_CLASS = 300

EPOCHS = 25
EPOCHS_HEAD = 8
EARLY_STOP_PATIENCE = 8

N_TRIALS = 3
SEEDS = [42, 123, 456]
SEED = 42

TEST_FRAC = 0.20
VAL_FRAC = 0.20

NUM_WORKERS = 4
TTA_PASSES = 3

HEAD_LR = 3e-4
BACKBONE_LR_MULT = 0.05
WEIGHT_DECAY = 1e-2
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TARGETS = {
    "accuracy": 0.95,
    "sensitivity": 0.95,
    "specificity": 0.90,
    "precision": 0.93,
}
