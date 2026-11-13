Biosignal-Based Human Intent Recognition Using Machine Learning

Decoding hand-gesture intent from surface EMG, from a custom wearable armband or public datasets,
with a focus on **how well decoders generalise to people they were not trained on**.

```
 wearable armband / public dataset
        │  raw multi-channel EMG (T × C)
        ▼
 band-pass 20–450 Hz + powerline notch  →  200 ms windows, 100 ms step
        │
        ├──► hand-crafted features (42 / channel)          ├──► raw windows (C × 200)
        │     statistical · FFT spectral · db4 wavelet     │
        │     → PCA / t-SNE / UMAP                         │
        │     → LDA · SVM · Random Forest                  │     → 1-D CNN · CNN-BiLSTM · Transformer
        ▼                                                  ▼
            evaluation: within-participant · leave-one-subject-out · LOSO + 1-rep calibration
```

## Project layout

```
hardware/
  emg_armband/emg_armband.ino   ESP32 firmware: 8 ch @ 1 kHz, binary packets with checksum
  record_session.py             guided gesture-recording app (prompts, labels, reps) → .npz
src/bsi/
  data/synthetic.py             realistic multi-participant sEMG simulator (no download needed)
  data/public.py                loaders: UCI EMG-for-gestures (36 ppl), NinaPro DB5 / DB2
  data/wearable.py              loader for sessions from your own armband
  preprocessing.py              filtering, windowing, rest-class balancing
  features/extract.py           time-domain, FFT and wavelet features (vectorised)
  reduction.py                  PCA summary, t-SNE / UMAP embeddings + trustworthiness
  models/classical.py           Scaler → PCA → LDA / SVM / RF pipelines
  models/deep.py                CNN, CNN-BiLSTM w/ attention, patch Transformer (PyTorch)
  evaluation.py                 within / LOSO / LOSO-calibration protocols, metrics
  viz.py                        all figures
scripts/run_pipeline.py         end-to-end run, driven by configs/default.yaml
docs/DATASETS.md                where to download the public datasets and how to lay them out
tests/                          8 unit tests (loaders on mock files, features, windowing)
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py                 # synthetic data: ~6 min for classical models on a laptop CPU; deep models take much longer on CPU (a GPU is recommended)
python scripts/run_pipeline.py --quick         # LDA + SVM + CNN only
pytest -q

# real data (see docs/DATASETS.md)
python scripts/run_pipeline.py --dataset uci --data-root data/EMG_data_for_gestures-master --out-dir results_uci
python scripts/run_pipeline.py --dataset ninapro --data-root data/ninapro_db5 --ninapro-fs 200 --out-dir results_db5

# your own wearable
python hardware/record_session.py --port /dev/ttyUSB0 --subject S01 --reps 6    # repeat per participant
python scripts/run_pipeline.py --dataset wearable --data-root data/wearable --out-dir results_wearable
```

Outputs go to `results/`: `summary.csv`, `per_subject_results.csv`, `run_info.json`, and `figures/`.

## Methods

**Signals.** Each participant performs 6 repetitions of 6 gestures plus rest (fist, wrist flexion/extension,
radial/ulnar deviation, extended palm), held for 3 s in random order. Windows must have a single label and lie inside
one repetition. Transition periods are discarded, so no window mixes two intents.

**Features (42 per channel, 336 total for 8 channels).**
* *Statistical / time domain:* MAV, RMS, variance, waveform length, zero crossings, slope-sign changes, Willison
  amplitude, skewness, kurtosis, Hjorth mobility & complexity, log-detector, AR(4) Yule-Walker coefficients.
* *FFT spectral:* log total power, mean / median / peak frequency, spectral entropy and spread, 5 relative band powers.
* *Wavelet:* 4-level db4 DWT. For each sub-band: relative energy, standard deviation and mean absolute coefficient value.
* Features are z-scored **within each participant**. This only uses unlabeled data from that person, so it is valid at
  deployment time, and it removes a lot of person-specific amplitude scaling.

**Dimensionality reduction.** PCA is used inside every classifier pipeline (keeping 95% of variance). t-SNE and UMAP are
used for visualisation and scored with *trustworthiness* (how well local neighbourhoods are preserved).

**Models.** Feature-based models: LDA (shrinkage), RBF-SVM and Random Forest. Deep models on raw windows: a 5-block
1-D CNN, a conv-stem + 2-layer BiLSTM with attention pooling, and a patch-embedding Transformer (3 layers, CLS token).
Deep models are trained with AdamW, one-cycle LR, label smoothing and early stopping. Their augmentations mimic
person-to-person variation: per-channel gain, noise and ±1 electrode rotation of the armband.

**Evaluation across participants.**
| Protocol | Train | Test | Answers |
|---|---|---|---|
| `within` | reps 1–4 of person *k* | reps 5–6 of person *k* | best case: a personally calibrated decoder |
| `loso` | all other people | person *k* | "plug-and-play" decoding for a new user |
| `loso_calib` | all others + rep 1 of person *k* | reps 2–6 of *k* | how far one short calibration closes the gap |

Reported: accuracy, balanced accuracy and macro-F1 for each participant, mean ± SD, and the **worst-participant accuracy**.
Worst-case accuracy matters more than the mean for an assistive device.

## Results (synthetic 10-participant dataset, classical models)

| Model | Within-participant | LOSO (new user) | LOSO + 1-rep calibration | Worst participant (LOSO) |
|---|---|---|---|---|
| LDA | 0.869 ± 0.068 | 0.610 ± 0.294 | 0.617 ± 0.279 | 0.214 |
| Random Forest | 0.840 ± 0.061 | 0.647 ± 0.299 | 0.660 ± 0.275 | 0.254 |
| **SVM (RBF)** | **0.911 ± 0.049** | **0.713 ± 0.214** | **0.738 ± 0.174** | **0.374** |

Dimensionality: 71 / 95 / 153 principal components keep 90 / 95 / 99% of the variance of the 336 features.
Trustworthiness of the 2-D embeddings: PCA 0.86, t-SNE 0.99.

Key observations
1. **Accuracy drops about 20 points when moving to an unseen person** (SVM 91% → 71%), and the spread across people grows
   four-fold. The t-SNE plot colored by participant shows why: windows cluster partly by *person*, not only by gesture.
2. Rest is always separable. Most cross-person confusions are between gestures that use overlapping muscle groups.
3. Adding one calibration repetition (~20 s of data) mostly helps the *worst* users (SVM worst case 0.37 → 0.46).
   Because the calibration data is only ~2% of the training set, giving it a higher sample weight or fine-tuning on it
   (deep models) is the natural next step.

> These numbers come from the built-in simulator, which models electrode shift, skin impedance, spectral differences,
> effort variation and noise. It is a test bed for the pipeline, not a claim about human data. Rerun on UCI / NinaPro
> (one command) to get reportable results. Published within-subject accuracy on NinaPro DB5 is typically around 70–90%
> depending on the exercise. UCI 6-gesture accuracy is usually higher.

## Figures
| | |
|---|---|
| `01_raw_emg.png` – 8-channel EMG with gesture labels | `02_spectra.png` – FFT spectrum per gesture |
| `03_pca_variance.png` – cumulative explained variance | `04_embeddings.png` – PCA / t-SNE (/ UMAP) by gesture and by participant |
| `05_protocols.png` – within vs cross-participant box plots | `06_per_subject.png` – accuracy for each participant |
| `07_confusion_*.png` – confusion matrices | `08_feature_importance.png` – top random-forest features |

## Extending
* **Inter-session drift:** add a "train day 1 / test day 2" protocol with GRABMyo.
* **Domain adaptation:** CORAL / adversarial (DANN) alignment of features between source people and the new user.
* **Real-time:** the 200 ms window / 100 ms step setting fits the ~300 ms controller-delay budget for prostheses.
  Add majority-vote smoothing over the last 3 decisions.
* **Other biosignals:** the same pipeline works for EEG/ECG/IMU. Change the band-pass and the frequency bands in `extract.py`.
