# Brain Tumor Classification

![Python](https://img.shields.io/badge/Python-FFD3D4?style=flat-square&logo=python&logoColor=775C56)
![PyTorch](https://img.shields.io/badge/PyTorch-F7BFC3?style=flat-square&logo=pytorch&logoColor=775C56)
![EfficientNet](https://img.shields.io/badge/EfficientNet_B3-FFEAEB?style=flat-square&logoColor=775C56)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-FFD3D4?style=flat-square&logoColor=775C56)

Four-class brain tumor classification from MRI — glioma, meningioma, pituitary, no tumor.

The question was whether heavier augmentation improves diagnostic accuracy over a plain baseline. It looked like it did. It didn't, and finding out why is most of what this repo is about.

## The leakage

My first version reported 97.2% accuracy for the augmented models against 93.5% for the baseline, and a significant t-test to go with it. All of that was wrong.

The bug was one line. To build the augmented arm I wrote:

```python
paths = X_data + X_data          # every image now appears twice
labels = list(np.tile(y_data, 2))
```

and then split on indices. Each image had two entries in the list, so one copy could land in train and the other in test. Roughly **60% of the test set was also in training** for every augmented model. The baselines used the path list once, so they had no leakage at all.

So the comparison was a clean model against a model that had memorized most of its own test set. The 3.7-point "improvement" was the leak.

| | leakage | reported accuracy |
|---|---|---|
| M1 baseline | none | 93.5% |
| M2 baseline | none | 93.8% |
| M3 SVM | none | 86.9% |
| M4 augmented | ~60% | 97.2% |
| M5 augmented | ~60% | 94.2% |
| M6 augmented | ~60% | 91.9% |
| M7 augmented | ~60% | 94.9% |

`tests/test_splits.py` reproduces the old split and asserts it leaked, so the number stays concrete instead of turning into a vague note about a past mistake.

**Corrected results are not in yet.** The code in this repo is fixed and tested, but I have not re-run the full sweep, so I am not publishing accuracy numbers I cannot stand behind. Expect the augmented models to land at or slightly above the baselines, not four points above.

## What changed

All splitting goes through `split_by_image()`, which raises if you hand it a list containing the same path twice. Augmentation and repeats apply to training indices only, after the split.

Two other things I had labeled wrong:

**"Synthetic" wasn't synthetic.** There was no generative model anywhere in the pipeline. It was the same images with stronger augmentation. I renamed those arms to "heavy aug," which is what they are.

**"TCIA" wasn't TCIA.** The loader only ever downloaded the Kaggle dataset. Models 2, 3, 5, 6 and 7 were labeled TCIA but trained on identical Kaggle data, which is why M1 and M2 scored within noise of each other — they were the same experiment run twice. The configs now say what they actually are: different backbones on one dataset.

The paired t-test also has to go. It ran on n=2 with models that aren't paired in any meaningful sense. If I want a significance test it needs to compare the same architecture across seeds.

## Layout

```
src/
  config.py      hyperparameters and metric targets
  splits.py      train/val/test splitting, with the duplicate guard
  data.py        dataset, transforms, loaders
  models.py      ResNet50, EfficientNet-B3, MobileNetV2 extractor
  train.py       two-phase fine-tuning, focal loss, early stopping
  evaluate.py    metrics, TTA, biopsy pre-screener
tests/
  test_splits.py the guards
run_experiments.py
```

## Running it

```bash
pip install -r requirements.txt
python run_experiments.py --smoke     # 2 configs, 1 seed, ~2 min
python run_experiments.py             # full sweep, ~50 min on a T4
pytest
```

Dataset downloads through `kagglehub` on first run, or is picked up from `/kaggle/input/` if you're in a Kaggle notebook.

## Method

Both CNNs train in two phases. The backbone freezes first so the new head can converge without wrecking the pretrained weights, then everything unfreezes with the backbone at 5% of the head's learning rate. Focal loss with label smoothing, since glioma and meningioma are the confusable pair and I didn't want the model coasting on the easy no-tumor class. Test-time augmentation averages softmax over three light augmentations.

Images run at 128px and 300 per class rather than full resolution and full dataset. That costs accuracy in absolute terms, but every arm pays the same cost, so the comparison between arms is still fair. It also keeps a 7-model, 3-seed sweep inside one Kaggle session.

## The biopsy screener

`BiopsyPreScreener` maps a prediction onto a triage level and a note. It's the interface piece of the project and it is a study aid, not a medical device. Nothing here is validated for clinical use and none of it should inform a real referral.

## Credits

I built and ran the modeling pipeline, data loading, model definitions, training loop, evaluation, and the screener and this repository is my code. The project was advised by **Jungsoo Han**.

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar.

## License

MIT.
