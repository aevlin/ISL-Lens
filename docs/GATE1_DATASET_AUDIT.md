# Gate 1 Dataset Audit

Status: initial audit complete, no dataset downloaded

Date: 2026-09-21

## Purpose

Gate 1 determines which datasets are suitable for ISL-Lens before any model training begins. The goal is not to collect the largest possible dataset immediately. The goal is to choose data that supports defensible isolated sign recognition, open-set rejection, leakage control, and signer-independent evaluation.

This audit is based on public dataset cards, papers, and dataset pages. All statistics must be revalidated after local acquisition because public pages may be incomplete, outdated, or inconsistent with actual files.

## Gate 1 Acceptance Criteria

Gate 1 is complete enough to move to Gate 2 when the project has:

- selected a primary isolated-sign dataset candidate;
- documented secondary datasets and why they are not primary;
- recorded source URLs, license status, vocabulary size, sample counts, signer metadata availability, and split availability;
- defined a machine-readable manifest schema;
- defined leakage-control rules;
- defined the first vocabulary strategy;
- identified unknown/OOV data requirements;
- documented unresolved risks before download.

## Candidate Dataset Summary

| Dataset | Best Use | Initial Recommendation |
| --- | --- | --- |
| INCLUDE | Primary isolated ISL recognition candidate | Use as first serious candidate after local verification |
| ISL Isolated 40 Words aggregate | Small prototype / pipeline validation | Use cautiously because it is derived from multiple sources |
| ISLRTC Indian Sign Language Dictionary / re-encodes | Vocabulary expansion candidate | Audit carefully; likely weak signer/split metadata |
| CISLR | Large-vocabulary word-level recognition research | Investigate access and metadata before use |
| ISL-CSLTR | Sentence-level / later translation work | Reserve for Gate 8, not isolated word model selection |
| ISL-Fingerspelling | Fingerspelling / temporal localization research | Separate subsystem, not primary word-sign recognition |
| IIITA-ROBITA ISL Gesture Database | Small historical benchmark | Not sufficient as primary dataset |
| Kaggle Indian Sign Language Video Dataset | Additional isolated videos | Candidate for augmentation only after provenance review |

## Primary Candidate: INCLUDE

Source:

- AI4Bharat INCLUDE page: https://github.com/AI4Bharat/sign-language.ai4bharat.org/blob/master/docs/INCLUDE.md
- Paper record: https://doi.org/10.1145/3394171.3413528

Publicly reported characteristics:

- isolated Indian Sign Language recognition dataset;
- 263 word-sign classes;
- 4,287 videos;
- around 0.27 million frames;
- INCLUDE-50 subset with 50 classes and 958 videos;
- 25 fps;
- 1920 x 1080 videos;
- average video length around 2.57 seconds;
- broad semantic categories such as greetings, pronouns, occupations, places, colors, objects, and others.

Strengths:

- directly aligned with isolated ISL recognition;
- large enough to support meaningful baselines;
- public research citation exists;
- INCLUDE-50 can support early fast experiments;
- vocabulary size aligns with the project goal of scaling beyond 100 signs.

Concerns to verify locally:

- exact license terms from the dataset distribution, not only the paper or secondary pages;
- whether signer identity is present for every video;
- whether official train/test splits are random or signer-independent;
- whether duplicate or near-duplicate samples exist;
- whether labels are normalized consistently;
- whether all files are downloadable and playable;
- whether any classes are visually too similar without enough samples;
- whether demographic/signer metadata is sufficient for signer-independent claims.

Initial decision:

INCLUDE should be the first primary dataset candidate for Gate 2 preprocessing work, but only after local manifest validation confirms license, file integrity, label list, and signer/split metadata.

## Secondary Candidate: ISL Isolated 40 Words Aggregate

Source:

- Hugging Face dataset card: https://huggingface.co/datasets/vidit031/isl-isolated-40words

Publicly reported characteristics:

- 642 isolated-sign clips;
- 40 target glosses;
- H.264 MP4 clips;
- height normalized to 480;
- approximately 30 fps;
- metadata includes word, normalized word, source dataset, original label, video path, signer, split, fps, resolution, duration, license, paper, repository, download URL, sha256, perceptual hash, quality score, duplicate status, and review status;
- aggregate from ISL500/ISL-DATA, INCLUDE, CISLR, and ISLRTC dictionary sources.

Strengths:

- small enough for quick local pipeline testing;
- metadata appears unusually useful for provenance, hashing, and signer-aware split experiments;
- vocabulary includes practical communication signs such as hello, thank you, yes, no, help, stop, family terms, places, food/drink, and question words.

Concerns:

- derived multi-source aggregate, not a single clean dataset license;
- some labels are aliases from upstream labels and may need manual review;
- only 40 words and 642 clips, so it is not enough for the final 100+ sign target;
- source-domain mixture may introduce dataset-specific bias;
- must respect each upstream license independently.

Initial decision:

Use only as a low-cost pipeline validation dataset or sanity-check dataset after license review. Do not treat it as the main accuracy benchmark.

## Vocabulary Expansion Candidate: ISLRTC Dictionary / Data.gov Re-encodes

Sources:

- Data.gov India resource: https://ap.data.gov.in/resource/indian-sign-language-dictionary-till-january-2024
- Hugging Face re-encode example: https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded

Publicly reported characteristics:

- Indian Sign Language Dictionary from ISLRTC / Government of India;
- thousands of short isolated sign videos for words, alphabets, and numbers;
- third-party re-encodes report around 75 GB after H.265 compression from an original around 200 GB;
- metadata completeness depends on the original source and re-encoding process.

Strengths:

- broad vocabulary potential;
- useful for future expansion beyond the initial dataset;
- official government-origin source.

Concerns:

- very large size, so it must not be downloaded before a focused acquisition plan;
- signer metadata and split support may be limited;
- dictionary videos may have single exemplars or inconsistent signer diversity;
- re-encoded copies may introduce compression artifacts;
- original data.gov terms must be checked directly, not inferred from third-party mirrors;
- dictionary signs may not provide enough signer variation for robust model training.

Initial decision:

Do not use as the first training dataset. Audit as a future vocabulary-expansion source after INCLUDE-based preprocessing and evaluation infrastructure exist.

## Large-Vocabulary Candidate: CISLR

Source:

- ACL Anthology paper page: https://aclanthology.org/2022.emnlp-main.707/

Publicly reported characteristics:

- word-level Indian Sign Language recognition corpus;
- around 4,700 words;
- video-based recognition;
- proposed in EMNLP 2022 with one-shot learning baselines.

Strengths:

- very large vocabulary;
- directly relevant to word-level ISL recognition;
- useful for future scaling research.

Concerns:

- access path, license, signer metadata, and file availability need verification;
- one-shot or low-resource setting may not match the first ISL-Lens training phase;
- may be too broad for initial controlled model comparison.

Initial decision:

Investigate further before adoption. It is not the first primary dataset until access, license, and metadata are verified.

## Sentence-Level Candidate: ISL-CSLTR

Source:

- Kaggle page: https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset

Publicly reported characteristics:

- sentence-level ISL corpus;
- 700 fully annotated videos;
- 18,863 sentence-level frames;
- 1,036 word-level images;
- 100 spoken-language sentences;
- 7 signers;
- CC BY-SA 4.0 license on the Kaggle page.

Strengths:

- relevant to future continuous sign language translation and sentence generation;
- includes sentence-level annotations;
- signer count is reported.

Concerns:

- not a replacement for isolated word recognition training;
- relatively small number of sentences;
- annotation structure and train/test split need local validation;
- sentence-generation claims must wait until paired labels are inspected.

Initial decision:

Reserve for Gate 8 sentence-generation investigation. It should not drive Gate 2 isolated-sign preprocessing decisions.

## Fingerspelling Candidate: ISL-Fingerspelling

Sources:

- Project page: https://kirandevraj.github.io/ISL-Fingerspelling/
- Hugging Face dataset card: https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling

Publicly reported characteristics:

- first continuous fingerspelling dataset for Indian Sign Language;
- 1,308 segments;
- 70.85 minutes;
- 14,685 character instances;
- includes word-level and character-level annotations;
- includes standard and signer-independent split files;
- project page states research-oriented data from public ISH News YouTube videos with CC BY-NC 4.0 overlap/context.

Strengths:

- unusually strong temporal annotation for fingerspelling;
- includes signer-independent split metadata;
- useful for future character-level or fingerspelling subsystem.

Concerns:

- fingerspelling is not the same task as isolated word-sign recognition;
- non-commercial license constraints may affect project usage;
- data source overlap with other corpora needs tracking.

Initial decision:

Keep separate from the primary isolated word-sign model. Consider later if ISL-Lens adds fingerspelling recognition.

## Historical Small Dataset: IIITA-ROBITA

Source:

- IIITA-ROBITA dataset page: https://robita.iiita.ac.in/dataset.php

Publicly reported characteristics:

- 23 isolated ISL gestures;
- RGB frame sequences;
- 30 fps;
- 320 x 240 resolution;
- constant background with varying illumination;
- password-protected distribution after release agreement.

Strengths:

- known historical ISL benchmark;
- controlled capture conditions may help sanity-check preprocessing.

Concerns:

- too small for the project goal;
- access requires agreement and password;
- constant background may not represent real webcam conditions;
- not enough vocabulary diversity for 100+ signs.

Initial decision:

Not a primary dataset. Use only if a small controlled benchmark is needed and institutional access is available.

## Additional Candidate: Kaggle Indian Sign Language Video Dataset

Source:

- Kaggle page: https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset

Publicly reported characteristics:

- word-level ISL videos in MP4 format;
- 60 unique signs;
- 60 videos per sign;
- additional still-sign videos;
- sourced from official Indian Sign Language YouTube channel;
- Kaggle page reports MIT license.

Strengths:

- useful sample count per class if provenance is clean;
- practical intermediate vocabulary size;
- may help with isolated word recognition experiments.

Concerns:

- exact signer diversity is unclear from the initial page;
- videos sourced from YouTube may have licensing/provenance questions despite Kaggle license;
- still-sign videos may bias the model if mixed with dynamic signs without care;
- official train/test splits and signer-independent splits need verification.

Initial decision:

Candidate for later augmentation, not first primary dataset.

## Recommended Initial Dataset Plan

1. Use INCLUDE as the primary isolated-sign candidate.
2. Use INCLUDE-50 first for fast preprocessing and baseline development if local metadata supports it.
3. Do not expand to 100+ classes until preprocessing, leakage checks, and baseline evaluation are working.
4. Keep the ISL Isolated 40 Words aggregate as a small pipeline sanity-check option only after license review.
5. Keep ISL-CSLTR separate for future sentence-generation work.
6. Keep ISL-Fingerspelling separate for future fingerspelling work.
7. Defer ISLRTC dictionary expansion until the project can handle large data and weak metadata responsibly.

## Required Manifest Fields

Every acquired sample must be represented by a manifest row with:

- dataset ID;
- dataset version;
- source URL;
- local relative path;
- media type;
- file size;
- checksum;
- perceptual hash if available;
- label/gloss;
- normalized label;
- vocabulary ID;
- signer ID if available;
- signer metadata if available;
- recording/session ID if available;
- source video ID if available;
- frame rate;
- duration;
- resolution;
- split assignment;
- split strategy;
- license;
- annotation type;
- quality status;
- duplicate status;
- review status;
- notes.

The first machine-readable schema is stored at:

```text
data/manifests/dataset_manifest.schema.json
```

The current candidate dataset registry is stored at:

```text
data/manifests/candidate_datasets.json
```

## Leakage-Control Rules

The split generator must eventually enforce:

- no same signer across signer-independent train/validation/test splits;
- no same recording/session across train/validation/test;
- no same source video across train/validation/test;
- no duplicate or near-duplicate videos across train/validation/test;
- no augmented copy crossing split boundaries;
- no threshold tuning on final test data;
- no hyperparameter selection on final test data.

If signer metadata is unavailable, the project must report that signer-independent evaluation is not supported for that dataset and must not imply real signer-independent performance.

## Initial Vocabulary Strategy

The initial vocabulary must be chosen from audited dataset labels, not manually invented from desired app features.

Recommended first vocabulary:

- Start with INCLUDE-50 if local metadata is valid.
- Prefer signs with enough samples per class and clear labels.
- Exclude labels requiring unresolved aliasing or ambiguous semantics.
- Mark all non-vocabulary gestures as OOV and require `UNKNOWN`.

Vocabulary records must eventually include:

- vocabulary ID;
- normalized gloss;
- display label;
- source dataset label;
- class index;
- inclusion reason;
- minimum sample count;
- known visually similar signs;
- supported/not supported status.

## Unknown/OOV Data Strategy

Open-set evaluation needs data beyond known-class validation clips. Candidate unknown sources include:

- held-out classes from the same dataset;
- unsupported signs from other ISL datasets;
- random non-sign hand motion collected under consent;
- resting-hand sequences;
- transition segments;
- poor tracking examples;
- incomplete signs.

Unknown data must be kept separate from final known-sign test data. Rejection thresholds must be selected on validation unknowns, then evaluated on held-out unknowns.

## Gate 1 Risks

- INCLUDE metadata may not support signer-independent splits.
- Public license claims may differ from actual dataset distribution terms.
- Dataset labels may not match the communication vocabulary desired by the app.
- Dictionary-scale datasets may have broad vocabulary but weak signer diversity.
- Aggregated datasets may mix licenses and domain styles.
- Sentence-level datasets may be too small for robust natural-language generation.
- Fingerspelling data may tempt scope expansion before isolated recognition is strong.

## Gate 1 Outcome

Gate 1 establishes a dataset-first plan but does not download data or train models.

Recommended next action:

Proceed to a controlled local acquisition of INCLUDE metadata and a small file-integrity audit only after the user approves download/access steps. Then generate the first real manifest from local files and verify whether signer-independent splitting is possible.

