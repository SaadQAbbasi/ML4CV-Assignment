# Person Search on PRW: detection + re-identification

**Student:** Saad Qasim Abbasi · ID `0001184186` · saadqasim.abbasi@studio.unibo.it

`main.ipynb` evaluates four person-search systems on the official PRW test split (2,057 queries, 6,112 gallery frames) with the official `eval_search_prw`. It is **inference only**. The trained weights are loaded from public Kaggle datasets. The submitted notebook already contains the outputs of a full run, so it can be read without being run.

- Kaggle notebook (public, inputs attached): https://www.kaggle.com/code/saadqasimabbasi/person-search-in-wild
- Code, training notebooks and retraining instructions: https://github.com/SaadQAbbasi/ML4CV-Assignment

## How to run

No accounts, tokens or secrets are needed.

1. Open the Kaggle notebook above and click **Copy & Edit**.
2. Settings: **Accelerator → GPU T4 ×1** (or P100). Internet is not required.
3. **Run All.** This takes about **30 min** on a T4.

By default only the final system (**Swin + DINO**) is evaluated live on the full test set. The other three systems load the saved outputs of an earlier full run of all four. The notebook prints the live metrics next to the saved ones, so you can see they match.

Optional settings (configuration cell, section 2.2):

- `RERUN_SYSTEMS`: the systems evaluated live. Set `RERUN_SYSTEMS = [s["name"] for s in SYSTEMS]` to re-evaluate all four (about 1 h 55 min).
- `QUICK_RUN = True`: a quick end-to-end test (a few minutes), with the live systems evaluated only on the first 300 gallery frames. Its numbers are **not** comparable to the reported results.

**Running the uploaded `main.ipynb` instead:** create a new Kaggle notebook, use *File → Import Notebook*, and attach these four datasets with *Add Input*:

| Input | Kaggle dataset |
|---|---|
| PRW dataset | `edoardomerli/prw-person-re-identification-in-the-wild` |
| Code | `saadqasimabbasi/prw-person-search-code` |
| Weights | `saadqasimabbasi/prw-person-search-weights` |
| Saved results | `saadqasimabbasi/prw-person-search-results` |

The notebook finds the inputs automatically.

## Results

Official `eval_search_prw`, PRW test set. **Detected** = gallery built from detector boxes (the real task). **GT** = ground-truth boxes (a perfect detector).

| System | mAP detected | Top-1 detected | mAP GT | Top-1 GT |
|---|---:|---:|---:|---:|
| ResNet baseline (crop) | 31.73 | 68.74 | 37.74 | 73.16 |
| Swin single-pass | 25.00 | 54.74 | 27.02 | 54.79 |
| Swin two-pass | 45.42 | 83.67 | 50.24 | 85.71 |
| **Swin + DINO (final)** | **54.07** | **86.49** | **59.47** | **88.09** |

**Final system:** a fine-tuned Swin-T Faster R-CNN detector (frozen during ReID training). Each detected person is cropped from the original frame and embedded by a fine-tuned **DINOv3 ViT-S/16**. The embedding concatenates the CLS token with the mean patch tokens of the upper and lower halves, projected to 256-d. Gallery boxes are ranked by cosine similarity to the query.

**Key findings**
- A dedicated crop encoder beats ReID on frozen detector features: **+20.4 mAP / +28.9 top-1** (single-pass → two-pass, same detector).
- Replacing the supervised Swin-T crop encoder with self-supervised DINOv3 adds **+8.6 mAP / +2.8 top-1**.
- Overall, the final system improves on the ResNet baseline by **+22.3 mAP / +17.7 top-1**.
