# Counterfactual Gender Bias in Vision-Language Models

B.Sc. (Hons) Computer Science dissertation project (University of Delhi, 2026) measuring gender bias in **SigLIP 2** and three small open vision-language models (VLMs) with **200 counterfactual prompt pairs**. The two prompts in each pair differ only in their gendered words (*she*/*he*, *her*/*his*, *woman*/*man*, …), and each pair is shown with a gender-neutral image. If a model's answers change when only the gender word changes, that change is evidence of bias.

Everything runs in one notebook, [`gender_bias_vlm.ipynb`](gender_bias_vlm.ipynb), on a single free-tier Colab T4 GPU.

## Method

**Data.** 200 prompt pairs adapted from [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) (Nangia et al., 2020), split evenly across four bias categories: *General* stereotypes, *Occupation*, *Leadership* and *Domestic / caregiving*. Every pair has one image with no people in it: workplaces and uniforms for occupations, emoji for emotions, and hands doing a task for domestic and general prompts.

**Representational bias (SigLIP 2).** Prompts and images are embedded with `google/siglip2-large-patch16-384`:

- **Bias score** = `1 − cos(she, he)`: how far apart the two prompts sit in embedding space.
- **Image alignment gap** = `cos(img, she) − cos(img, he)`: whether the neutral image sits closer to one gender.

Embedding-level scores do not reliably predict how biased a model's outputs are (Goldfarb-Tarrant et al., 2021), so these are reported alongside the behavioural results, not as a stand-in for them.

**Behavioural bias (VLMs).** Each VLM sees the neutral image with each prompt and answers three questions: the bias category (4 options), the subcategory (4 options), and a sentiment score in [−1, 1].

- **Mean sentiment gap** = mean of `sentiment(she) − sentiment(he)`: the *direction* of bias.
- **Mean absolute gap** = mean of `|sentiment(she) − sentiment(he)|`: the *size* of bias, without she- and he-favouring pairs cancelling out. This is the counterfactual token fairness gap (Garg et al., 2019); with one deterministic score per prompt it also equals the average individual-fairness distance of Huang et al. (2020).
- **Changed / she-favoured / he-favoured**: share of pairs whose sentiment score changed, and in which direction.
- **Classification agreement**: share of pairs where both prompts get the same category.
- **Category accuracy**: share of answers that match the labelled category, for each gender separately.

**Analysis.** Every metric is reported for all 200 pairs and separately for each category (4 groups of 50 with no overlap):

- a two-sided Wilcoxon signed-rank test that keeps zero gaps (Pratt, 1959), and a sign test on the direction of the pairs that changed, both Holm-corrected;
- 95% bootstrap confidence intervals (10,000 resamples) and Cohen's d_z;
- a Kruskal–Wallis test of whether the gap differs between categories;
- a split-half check: two halves with the same category mix and no pairs in common should agree;
- the number of pairs needed for 80% power at benchmark effect sizes, for planning a larger study.

| Model | Role | Parameters | Loaded as |
|---|---|---|---|
| [SigLIP 2 large/16-384](https://huggingface.co/google/siglip2-large-patch16-384) | embeddings | 882M | float16 |
| [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | VLM judge | 256M | 4-bit NF4 |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | VLM judge | 3B | 4-bit NF4 |
| [Gemma-4-E2B-it](https://huggingface.co/google/gemma-4-e2b-it) | VLM judge | 2B effective | 4-bit NF4 |

## Data layout

```
data/
├── pairs_BALANCED_200.xlsx   # sheet "Sheet1 (2)"
└── images/                   # one image per row, named by row index
    ├── 0.jpg
    ├── 1.png
    └── ... 199.jpeg
```

Spreadsheet columns:

| Column | Content |
|---|---|
| `GROUP A` | she-prompt |
| `GROUP B` | he-prompt (same text with the gender swapped) |
| `prompt_category` | `General`, `Occupation`, `Leadership` or `Domestic` |

Images can be `.jpg`, `.jpeg`, `.png` or `.webp`. A row without an image runs text-only.

**Licensing.** CrowS-Pairs is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), so a published copy of the spreadsheet must keep that licence and credit Nangia et al. (2020). Check each image's licence before publishing the images.

## Running

**Google Colab (recommended)**

1. Upload the notebook to Colab and select a **T4 GPU** runtime.
2. Put the spreadsheet and `images/` in a Google Drive folder, and set `DATA_DIR` in cell 1.2 to that folder.
3. Accept the [Gemma-4 licence](https://huggingface.co/google/gemma-4-e2b-it) and add your Hugging Face token as a Colab secret named `HF_TOKEN`.
4. Run cell 1.1, restart the runtime, then run everything from 1.2 down.

For a first check, set `QUICK_TEST = True` in cell 1.2. It runs 8 pairs in about 5 minutes and saves to a separate `quick_test` folder. If it finishes without errors, set it back to `False` for the full run.

**Locally (NVIDIA GPU with about 10 GB of VRAM or more)**

```bash
pip install -r requirements.txt
huggingface-cli login
jupyter notebook gender_bias_vlm.ipynb
```

Skip cell 1.1 when running locally.

A full run makes about 3,600 VLM calls (200 pairs × 2 genders × 3 questions × 3 models). Each model's outputs are saved as soon as it finishes, so an interrupted run picks up where it stopped.

## Outputs

Everything is written to `outputs/` (on Colab: `MyDrive/bias_pipeline_out_corrected/`):

| File | Content |
|---|---|
| `vlm_records_all.csv` | Every VLM answer: raw text, parsed value, per-pair gap |
| `summary_siglip.csv`, `summary_vlm.csv` | Metrics for all pairs and for each category |
| `stats_sentiment_gap.csv` | Wilcoxon and sign-test p-values (Holm-adjusted), bootstrap CIs and d_z per model and category |
| `stats_category_differences.csv` | Kruskal–Wallis tests: does the gap differ between categories? |
| `split_half.csv` | Results in two independent halves of the data |
| `stats_image_alignment_gap.csv` | Tests on SigLIP 2's image alignment gap |
| `power_planning.csv` | Pairs needed for 80% power at benchmark effect sizes |
| `*.html` | Interactive figures |
| `run_info.json` | Library versions, model revisions and settings for this run |

## Changes from the dissertation version

This version fixes bugs and methodological problems in the notebook used for the dissertation, so its numbers will differ:

- **Category parsing.** The index parser's regular expressions were double-escaped, so replies like `"2"` were never matched and every category answer fell back to option 0. Both genders therefore always got the same category, and the reported 100% classification agreement was an artifact. Answers are now parsed correctly, and anything unreadable is recorded as missing instead of defaulting to 0.
- **Missing answers.** Unparseable sentiment replies used to become 0.0, and inference errors became "category 0, sentiment 0". Both made pairs look unbiased. They are now excluded from each metric and counted in a parse-quality table.
- **Nested subsets replaced by categories.** The dissertation compared the first 50, 100, 150 and 200 pairs. The spreadsheet is sorted by category, so each larger subset just added a new category, and because the subsets overlap they were not independent evidence. Results are now reported for all pairs and for each category separately, with a test of whether categories differ and a split-half stability check.
- **Signed gap only.** A signed average lets pairs that favour *she* and pairs that favour *he* cancel out. The mean absolute gap and the share of changed pairs are now reported as well.
- **Zero gaps in the Wilcoxon test.** SciPy's default drops pairs with a zero gap, which is most pairs here; Pratt's method keeps them, and a sign test is added.
- **Effect sizes.** The dissertation describes a paired Cohen's d but reports values from a different formula (0.070 / 0.031 / 0.033 at n = 200). The paired d_z from the same run is about 0.14 / 0.09 / 0.14 (SmolVLM / Qwen / Gemma). The notebook now computes d_z directly.
- **Power analysis.** Power computed from the observed effect adds no information beyond the p-value (Hoenig & Heisey, 2001). The notebook instead reports the pairs needed at benchmark effect sizes.
- **Confidence-interval chart.** The CI bounds were passed to Plotly as error-bar *lengths*, which drew the intervals in the wrong place.
- **Radar and bubble charts removed.** Min–max scaling across three models showed rankings, not sizes, and mixed metrics that point in opposite directions.
- **Efficiency and reproducibility.** Each VLM now runs once instead of once per subset size. Raw outputs are saved to CSV instead of pickle, and library versions and model revisions are recorded.

## Limitations

- Gender is treated as a binary *she/he* swap. Non-binary identities and other demographic attributes are not covered.
- **Benchmark validity.** CrowS-Pairs has documented problems, including pairs that do not clearly express a stereotype (Blodgett et al., 2021). Some sentences are ungrammatical and some pairs swap several gendered words at once. Six pairs (#21, #23, #69, #159, #161, #163) contain typing errors in the swap, for example "Men do not work as hard as men"; re-analysing without them changes no conclusion.
- **Model-as-judge ratings.** Sentiment comes from each model rating the text itself, and such ratings shift with prompt wording and scale design (Evaluating Scoring Bias in LLM-as-a-Judge, 2025). Only one wording per question was used, and most pairs got the same score for both genders.
- **No embedding baseline.** Changing any single word moves an embedding, so the SigLIP 2 bias score should be compared with pairs that swap a non-gendered word before it is read as gender bias.
- About 50 pairs per category gives wide confidence intervals.
- The VLMs run in 4-bit quantisation, which can behave differently from full precision, and only small models (≤3B parameters) were tested.

## References

- Blodgett, S. L., Lopez, G., Olteanu, A., Sim, R., & Wallach, H. (2021). Stereotyping Norwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets. *ACL 2021*.
- Evaluating Scoring Bias in LLM-as-a-Judge (2025). arXiv:2506.22316.
- Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., & Beutel, A. (2019). Counterfactual Fairness in Text Classification through Robustness. *AIES 2019*.
- Goldfarb-Tarrant, S., et al. (2021). Intrinsic Bias Metrics Do Not Correlate with Application Bias. *ACL 2021*.
- Hoenig, J. M., & Heisey, D. M. (2001). The Abuse of Power: The Pervasive Fallacy of Power Calculations for Data Analysis. *The American Statistician*, 55(1).
- Huang, P.-S., et al. (2020). Reducing Sentiment Bias in Language Models via Counterfactual Evaluation. *Findings of EMNLP 2020*.
- Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. *EMNLP 2020*.
- Pratt, J. W. (1959). Remarks on Zeros and Ties in the Wilcoxon Signed Rank Procedures. *Journal of the American Statistical Association*, 54(287).
- Xiao, Y., et al. (2024). GenderBias-VL: Benchmarking Gender Bias in Vision Language Models via Counterfactual Probing. arXiv:2407.00600.

## Author

Kiran, B.Sc. (Hons) Computer Science, University of Delhi.

Presented as a poster at the 1st National Young Scholars' Conference (NYSC 2026), University of Delhi.
