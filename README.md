# Target-Level Sentiment in Turkish Reviews with Conflicting Opinions

Term project for **CSE 654 Natural Language Processing**, Gebze Technical University (Fall 2026). Instructor: Dr. Salih Sarp.

A single review often says different things about different targets: *"The hotel was great **but** the location was terrible."* We study how well fine-tuned Turkish encoders handle these **conflicting** reviews, what shortcuts they rely on, and whether large language models make the same mistakes.

## Research Questions

| | Question | Status |
|---|---|---|
| **RQ1** | Do fine-tuned encoders score lower on reviews whose targets have opposite sentiments? | Done |
| **RQ2** | Which shortcuts cause the errors (counterfactual "trap" tests), and can targeted training augmentation reduce them? | Conjunction test done; others in progress |
| **RQ3** | Do LLMs (zero-/few-shot) show the same weakness? | Planned |

## Data and Results (Google Drive)

The dataset, the fixed split, the trained models and all prediction files are in a shared Drive folder (too large for GitHub):

**[cse654_absa — Google Drive folder][drive]**

| Path in the folder | Content |
|---|---|
| `absa_10k_final.csv` | Dataset: 9,999 reviews, 27,290 target–polarity pairs. Columns: `id`, `text`, `target` (list), `polarity` (list), `category` |
| `splits/split_ids.csv` | Fixed train/dev/test split (`id` → `split`). Used by every model and by the LLM experiments |
| `sonuclar/` | Test predictions (`*_test_tahminleri.csv`) and metric tables for each model |
| `sonuclar/2_baglac_berturk/` | Conjunction trap test: per-seed predictions and the seed-level summary (`sonuc_seed_bazinda.csv`) |
| `berturk_final/`, `modernbert_tr_final/`, `xlmr_final/` | Fine-tuned baseline models (seed 42) |
| `berturk_baglac_seed/` | BERTurk models for seeds 43–45 (baseline and conjunction-augmented) |
| `duzeltme_baglac/` | Conjunction-augmented BERTurk, seed 42 |
| `artirma/` | Augmented training data (one subfolder per augmentation) |
| `*_checkpoints/` | Training checkpoints |

## Dataset

Human-annotated Turkish aspect-based sentiment dataset built in an earlier undergraduate project. Each review has one or more targets, each labelled `positive`, `negative` or `neutral`. Repeated targets and surface forms are kept as written in the text.

Each review falls into exactly one group:

| Group | Definition | Reviews |
|---|---|---:|
| Single-target | One target | 792 |
| Same-sentiment | Several targets, all with the same polarity | 5,014 |
| Neutral-mixed | Several targets, neutral mixed with one other polarity | 2,233 |
| **Conflicting** | At least one positive **and** one negative target | **1,960** |

## Method

- **Task:** the model reads `(review, target)` as a sentence pair and predicts one of three labels.
- **Split:** review-level (a review never appears in two splits; duplicate texts go to the same side). Each group is split 80/20 train/test and 10% of train is held out as dev. The test set is frozen.

  | Split | Reviews | Pairs |
  |---|---:|---:|
  | Train | 7,198 | 19,715 |
  | Dev | 800 | 2,168 |
  | Test | 2,001 | 5,407 |

- **Models:** [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased), [ModernBERT-TR](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base), [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base).
- **Training (identical for all models):** LR 2e-5, batch 32, 4 epochs, 10% warmup, weight decay 0.01, fp16, max length = 99.9th percentile of token lengths (rounded up to 32). The best epoch is chosen by **dev macro-F1**, never on test. Colab T4 GPU.
- **Metrics:** pair-level accuracy and macro-F1, plus a stricter review-level score: **all targets correct** (a review counts only if every target in it is right). Reported overall and per group.

## Results

### RQ1 — Baselines (test set, seed 42, %)

| Model | Accuracy | Macro-F1 | All targets correct |
|---|---:|---:|---:|
| BERTurk | 87.8 | 86.0 | 76.0 |
| ModernBERT-TR | 87.7 | 85.9 | 76.5 |
| XLM-R | 86.8 | 85.1 | 74.4 |

**By group** (accuracy / all targets correct):

| Group (reviews / pairs) | BERTurk | ModernBERT-TR | XLM-R |
|---|---:|---:|---:|
| Single-target (159 / 159) | 92.5 / 92.5 | 91.2 / 91.2 | 85.5 / 85.5 |
| Same-sentiment (1,003 / 2,818) | 93.4 / 87.3 | 93.0 / 86.7 | 92.1 / 84.8 |
| Neutral-mixed (447 / 1,257) | 75.4 / 48.3 | 75.5 / 50.6 | 75.4 / 49.9 |
| **Conflicting (392 / 1,173)** | **87.0 / 71.9** | **87.7 / 73.7** | **86.6 / 71.2** |

**BERTurk, 4 seeds (42–45), mean ± std:** accuracy 87.75 ± 0.39, macro-F1 86.08 ± 0.47, same-sentiment accuracy 93.26 ± 0.37, conflicting accuracy 86.72 ± 0.67, conflicting all-correct 70.92 ± 1.10.

**Findings**

- The three encoders are within about one point overall.
- Going from same-sentiment to conflicting reviews, all-targets-correct drops **13–15 points** and macro-F1 drops **17–21 points** for every model.
- The gap is not a length effect: length-matched same-sentiment accuracy is practically unchanged (BERTurk 93.4 → 93.4).
- Conflicting reviews **without** an explicit contrast conjunction are much harder (79.6–82.8% accuracy) than those with one (90.3–90.4%).
- 68–74% of errors on conflicting reviews copy the label of **another target in the same review**.
- Neutral-mixed reviews are the hardest group overall (~75% accuracy, ~50% all-correct).

### RQ2 — Conjunction trap test (BERTurk, 4 seeds, mean ± std, %)

- **Trap set:** 265 conflicting test reviews (761 pairs) that contain a contrast conjunction (*ama, fakat, ancak, lakin, oysa, ne var ki*). The conjunction is deleted; targets and labels stay the same.
- **Control:** a random function word outside the targets is deleted instead (640 pairs).
- **Fix:** conjunction-free copies of the training reviews that contain a conjunction are added to training (+4,917 pairs). Dev and test are unchanged.

| | Baseline | + Conjunction augmentation |
|---|---:|---:|
| Predictions changed when conjunction is removed | 5.88 ± 0.27 | 4.11 ± 0.67 |
| Correct → wrong after removal | 4.40 ± 0.17 | 2.60 ± 0.53 |
| Predictions changed, control deletion | 1.45 ± 0.15 | 1.99 ± 0.55 |
| Test accuracy (all) | 87.75 ± 0.39 | 87.37 ± 0.63 |
| Conflicting all-correct | 70.92 ± 1.10 | 68.69 ± 1.80 |

Removing the conjunction changes about 4× more predictions than the control deletion, so the models rely on the conjunction as a cue. The augmentation lowers this sensitivity in all 4 seeds, but does not improve test accuracy.

## Repository Structure

| Notebook | Content |
|---|---|
| `01_berturk_baseline.ipynb` | Builds and saves the split, trains BERTurk, group-level evaluation and error analysis |
| `02_modernbert_xlmr_baseline.ipynb` | Trains ModernBERT-TR and XLM-R on the same split and settings, compares all three models |
| `03_conjunction_trap_berturk_4seeds.ipynb` | Conjunction trap test, conjunction augmentation, BERTurk with 4 seeds |

## How to Run

1. Copy the [Drive folder][drive] to your own Drive as `MyDrive/cse654_absa` (the notebooks write results into this folder, so they need write access).
2. Open a notebook in Google Colab and select a **T4 GPU** runtime.
3. Run all cells. Dependencies are installed in the notebook (`transformers`, `accelerate`, `datasets`, `scikit-learn`).
4. Run the notebooks in order (01 → 02 → 03). Notebook 01 saves `splits/split_ids.csv` and later notebooks reuse it. Models that already exist in the folder are not retrained.

## Roadmap

- [x] Dataset cleaning and fixed review-level split
- [x] BERTurk, ModernBERT-TR, XLM-R baselines and group-level analysis
- [x] Conjunction trap test and conjunction augmentation (BERTurk, 4 seeds)
- [ ] Seeds 43–45 for ModernBERT-TR and XLM-R
- [ ] More trap tests (e.g. flipping a target's sentiment, swapping target positions) and matching augmentations
- [ ] LLM comparison on the same test set (zero-/few-shot)
- [ ] Final report and demo (December 2026)

## Team

Özcan Akay · Gülbeyaz Altuntaş · Umut Başaran · Yağmur Gül · Harun Kaya

[drive]: https://drive.google.com/drive/folders/1cdtRWxylkl1HbBVxQVcSxAbf7DVI12bF
