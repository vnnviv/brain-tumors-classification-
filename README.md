# Brain Tumor Classification

![Python](https://img.shields.io/badge/Python-FFD3D4?style=flat-square&logo=python&logoColor=775C56)
![PyTorch](https://img.shields.io/badge/PyTorch-F7BFC3?style=flat-square&logo=pytorch&logoColor=775C56)
![EfficientNet](https://img.shields.io/badge/EfficientNet_B3-FFEAEB?style=flat-square&logoColor=775C56)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-FFD3D4?style=flat-square&logoColor=775C56)

The question was whether heavier augmentation improves diagnostic accuracy over a plain baseline. It looked like it did. It didn't, and finding out why is most of what this repo is about.
 not four points above.


## Pipeline 

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

## Running 

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

## The biopsy Screening 

`BiopsyPreScreener` maps a prediction onto a triage level and a note. It's the interface piece of the project and it is a study aid, not a medical device. Nothing here is validated for clinical use and none of it should inform a real referral.

## Credits

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar.

## License

MIT.
