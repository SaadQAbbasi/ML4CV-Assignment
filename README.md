# Person Search on PRW: Swin-T detector + DINOv3 ReID

Given a crop of a person, **find every other appearance of that person** in 6,112 full frames from six campus cameras (the [PRW dataset](https://github.com/liangzheng06/PRW-baseline)). A system has to *detect* every person in every frame and *re-identify* them, ranking the detections by their similarity to the query.

The final system pairs a fine-tuned **Swin-T Faster R-CNN** detector with a separate **DINOv3 ViT-S/16** ReID encoder. It reaches **54.1 mAP / 86.5 top-1** on the PRW test set with the official `eval_search_prw`, which is in the range of published end-to-end methods such as COAT (≈ 53 / 87).

▶ **[Run it on Kaggle](https://www.kaggle.com/code/saadqasimabbasi/prw-search)**: no setup, no secrets. Click *Copy & Edit* and then *Run All*. [`main.ipynb`](main.ipynb) contains the full run with all outputs. If GitHub does not display it, [view it on nbviewer](https://nbviewer.org/github/SaadQAbbasi/ML4CV-Assignment/blob/main/main.ipynb).

## The final pipeline

![Swin + DINO pipeline](docs/diagrams/pipeline_4.png)

1. **Detect.** The Swin-T Faster R-CNN (512×768) is fine-tuned for detection only, then frozen. It keeps boxes with confidence ≥ 0.5.
2. **Re-crop.** Each box is cut out of the *original-resolution* frame and resized to 256×128.
3. **Embed.** The DINOv3 ViT-S/16 (last 4 blocks fine-tuned with SupCon) produces `[CLS | mean of upper-half patches | mean of lower-half patches]`, which is projected to an L2-normalised 256-d embedding.
4. **Rank.** Gallery detections are ranked by cosine similarity to the query.

## Results

Four systems form an ablation chain. All results use the official `eval_search_prw` on the 2,057 PRW test queries.

| System | Gallery ReID features | mAP | Top-1 |
|---|---|---|---|
| ResNet baseline | jointly trained ResNet-50 Faster R-CNN, crops re-encoded by the same backbone | 31.7 | 68.7 |
| Swin single-pass | RoIAlign on the **frozen detector features** + trained ReID head | 25.0 | 54.7 |
| Swin two-pass | **re-cropped pixels** → separate ImageNet Swin-T | 45.4 | 83.7 |
| **Swin + DINO (final)** | re-cropped pixels → **self-supervised DINOv3 ViT-S/16** | **54.1** | **86.5** |

![Official mAP and top-1 per system](docs/images/results.png)

*detected* = the gallery is the detector's boxes (the real task). *gt* = the gallery is the ground-truth boxes, i.e. a perfect detector.

**Main findings**
- **Re-cropping beats reusing detector features (+20.4 mAP).** With the same detector, a dedicated encoder on the original pixels is far better than ReID on the frozen detection features. Those features are trained to make all people look like "person", so they lose identity detail.
- **A self-supervised backbone helps (+8.6 mAP).** Swapping the supervised Swin-T crop encoder for DINOv3, with everything else unchanged, gives the largest gain after the architecture change.
- **Upper/lower pooling beats the plain CLS token (+1.0 to 1.2 mAP).** In a pedestrian crop the top half is mostly the upper body and the bottom half the legs. Pooling them separately keeps the "top colour vs bottom colour" cue. Averaging *all* patches gave no gain.
- **ReID, not detection, is the bottleneck.** A perfect detector would add only about 5 mAP.

## Qualitative results

The same query for all four systems: top-5 results, green = correct, red = wrong. The query person is partly occluded by a bicycle. The single-pass model returns five wrong people in similar light clothing, while the two-pass and DINO systems find her at rank 1.

![Top-5 retrievals for one query](docs/images/query_example.png)

**Embedding space.** t-SNE of 20 test identities × 8 appearances (one colour per person). The re-cropping systems form tight, separate clusters, and DINO's are the tightest.

![t-SNE of the embeddings](docs/images/tsne.png)

<details>
<summary>Query-vs-gallery cosine similarity matrices</summary>

![Cosine similarity matrices](docs/images/similarity.png)

</details>

The notebook contains the full discussion, nine more query examples and the limitations.

## Repository layout

```
main.ipynb          the submitted notebook: evaluation, ablation, qualitative results (with outputs)
src/                the project modules (model, data, losses, training and evaluation code)
training/           the four Kaggle notebooks that produced the published weights
tests/              unit tests (unittest)
tools/              prepare_kaggle_datasets.py: builds the Kaggle code/weights datasets (author only)
docs/diagrams/      pipeline diagram of each system (draw.io source + PNG)
docs/images/        figures from main.ipynb used in this README
requirements.txt    packages for running the code locally
```

`src/` modules:

```
eval_function.py           official eval_search_prw (provided with the assignment, unmodified)
official_evaluation.py     adapter: converts project outputs into eval_search_prw inputs
prw_evaluation.py          gallery detection/embedding, query embedding, in-house ranking, detector P/R
final_comparison.py        model loading, diagnostic sample and plots used by main.ipynb
prw_dataset.py             PRW annotation/frame loading, crop preprocessing, P×K identity sampler
retrieval_diagnostics.py   detector precision/recall matching
model.py                   Faster R-CNN person-search model (ResNet-50 / Swin-T backbone)
backbone.py, fpn.py, rpn.py, roi_heads.py   detector components
swin_reid.py               Swin single-pass (scene RoI head) and two-pass (crop encoder) ReID
dino_reid.py               detector + crop-encoder wrapper (DinoPersonSearch) and the crop training epoch
dinov3_reid.py             DINOv3 crop encoder with CLS / patch-region pooling
metric_loss.py             triplet, contrastive and supervised NT-Xent (SupCon) losses
training.py                joint training/validation loops, W&B run setup, AMP helpers
detection_training.py      detection-only training epoch and AP50/AP75 evaluation
validation_retrieval.py    retrieval mAP on the validation frames (checkpoint selection)
offline_mining.py          P×K sampler with optional hard-identity mining
```

## How to run `main.ipynb`: on Kaggle

The notebook is designed to run **on Kaggle** and needs no accounts, tokens or secrets.

**Public Kaggle notebook (inputs already attached):** https://www.kaggle.com/code/saadqasimabbasi/prw-search

1. Open the link above and click **Copy & Edit**.
2. Settings: **Accelerator → GPU T4 ×1** (or P100). Internet is not required.
3. **Run All.** This takes about 30 min on a T4. By default only the final system (Swin + DINO) is evaluated live on the full test set. The other three systems load the saved outputs of a full run of all four systems (the *Saved results* input). To re-evaluate them too, add their names to `RERUN_SYSTEMS` in section 2.2 (about 28 min each, 1 h 55 min for all four).

The submitted `main.ipynb` already contains every output of that full run, so it can be read without being run.

**Quick run (optional).** To check that the notebook runs end to end without waiting for the full evaluation, open the configuration cell in section 2.2, set `QUICK_RUN = True` and then *Run All*. Every live system is evaluated on only the first `QUICK_GALLERY_FRAMES = 300` test frames, and every section still produces its tables and figures. The quick-run mAP / top-1 are **not** comparable to the reported results. Those come from the full run with `QUICK_RUN = False`, which is the default.

To start instead from `main.ipynb`: create a new Kaggle notebook, use *File → Import Notebook*, and attach these five public datasets with *Add Input*:

| Input | Kaggle dataset | Content |
|---|---|---|
| PRW dataset | [`edoardomerli/prw-person-re-identification-in-the-wild`](https://www.kaggle.com/datasets/edoardomerli/prw-person-re-identification-in-the-wild) | frames, annotations, queries |
| Code | [`saadqasimabbasi/prw-person-search-code`](https://www.kaggle.com/datasets/saadqasimabbasi/prw-person-search-code) | the `src/` modules + a DINOv3 code snapshot |
| Weights | [`saadqasimabbasi/prw-person-search-weights`](https://www.kaggle.com/datasets/saadqasimabbasi/prw-person-search-weights) | the five trained checkpoints |
| Saved results | [`saadqasimabbasi/prw-person-search-results`](https://www.kaggle.com/datasets/saadqasimabbasi/prw-person-search-results) | metrics, top-5 retrievals of the first 100 queries and diagnostic embeddings from the full run of all four systems |
| PRW test archive (optional, faster) | [`saadqasimabbasi/prw-person-search-test-archive`](https://www.kaggle.com/datasets/saadqasimabbasi/prw-person-search-test-archive) | the PRW test split packed in one file, unpacked to local disk at the start: Kaggle can be very slow to read the ~14,000 small PRW files one by one |

The notebook finds the inputs automatically, wherever Kaggle mounts them under `/kaggle/input`. Results (CSV, JSON, PNG) are written to `/kaggle/working/final_prw_comparison`.

## Model weights

The weights dataset contains:

| File | System | Produced by |
|---|---|---|
| `resnet_baseline.pt` | ResNet baseline (joint detection + ReID) | `training/01_joint_person_search.ipynb`, `CONFIG = "resnet_baseline"` |
| `swin_detector.pt` | Swin-T detector shared by the three Swin systems | `training/02_swin_detector.ipynb` |
| `swin_single_pass.pt` | Swin single-pass (frozen detector + scene RoI ReID head) | `training/03_swin_reid.ipynb`, `STRATEGY = "crop_to_scene"` |
| `swin_two_pass.pt` | Swin two-pass (frozen detector + Swin-T crop encoder) | `training/03_swin_reid.ipynb`, `STRATEGY = "crop_to_crop"` |
| `swin_dino_reid.pt` | DINOv3 ReID encoder (final system) | `training/04_dinov3_reid.ipynb` |

Each file is the exact W&B artifact version listed in the `main.ipynb` manifest, with the optimizer state removed.

## Retraining the models

Not needed to run `main.ipynb`: the trained weights are already published. To rebuild them, run the notebooks in `training/` on a Kaggle T4. Notebooks 01 and 03 are run twice, with a different setting at the top.

| Run | Notebook and setting | Starts from | Produces | Time (T4) |
|---|---|---|---|---|
| 1 | `01_joint_person_search`, `CONFIG = "resnet_baseline"` | ImageNet | `resnet_baseline.pt` (ResNet baseline) | ≈ 3.1 h |
| 2 | `01_joint_person_search`, `CONFIG = "joint_swin_t_triplet"` | ImageNet | `joint_swin_t_triplet.pt` (intermediate, used by runs 3 and 6) | ≈ 2.6 h |
| 3 | `02_swin_detector` | run 2 | `swin_detector.pt` (detector used in `main.ipynb` by runs 4, 5 and 6) | ≈ 4 h |
| 4 | `03_swin_reid`, `STRATEGY = "crop_to_scene"` | run 3, frozen | `swin_single_pass.pt` (Swin single-pass) | ≈ 5.3 h |
| 5 | `03_swin_reid`, `STRATEGY = "crop_to_crop"` | run 3, frozen | `swin_two_pass.pt` (Swin two-pass) | ≈ 2.9 h |
| 6 | `04_dinov3_reid` | pretrained DINOv3 ViT-S/16 | `swin_dino_reid.pt` (Swin + DINO) | ≈ 2.7 h |

**Which detector is used where.** The ResNet baseline uses its own ResNet-50 detector from run 1. There are two Swin-T detectors: the run-2 detector, trained together with re-identification at 256×384, and the run-3 detector, which starts from it and is trained for detection only at 512×768, so it is better. **At inference in `main.ipynb`, all three Swin systems (runs 4, 5 and 6) use the same run-3 detector.**

**Why run 6 lists run 2.** The DINOv3 model was trained before run 3 existed. During training it learns only from ground-truth person crops, with no detector. After each epoch it is checked on the validation frames, where the run-2 detector finds the people; the epoch with the best validation mAP is kept. So run 2 only decides which epoch is saved and never changes what the model learns.

To pass a run's output to a later run, use *Add Input → Notebook Output* in the later notebook. Run 6 also needs Meta's `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, from the [DINOv3 repository](https://github.com/facebookresearch/dinov3) (DINOv3 license), uploaded as a private Kaggle dataset. Results can differ slightly between runs, because GPU training is not bit-wise deterministic.

## Running locally (optional)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests     # unit tests, no data needed
```

To run a training notebook outside Kaggle, point `PRW_INPUT_ROOT` at a folder that contains the PRW dataset, `src/` and the checkpoints, and set `PRW_WORKING` to an output folder.
