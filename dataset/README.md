# Authorized MaizeMask v1.0 data

This directory is tracked so the expected dataset location is visible after a
GitHub clone. The dataset itself is not in this repository.

## Dataset download

The canonical dataset archive will be published on Zenodo. Replace these
placeholders before making the repository public:

- Zenodo record or DOI: `REPLACE_WITH_ZENODO_DATASET_URL`
- Optional Google Drive mirror: `REPLACE_WITH_PUBLIC_GOOGLE_DRIVE_URL`

Zenodo should remain the citable source of record. A Google Drive URL, when
provided, is only a download mirror and must serve the same versioned archive.

After receiving an authorized MaizeMask v1.0 release from the authors, place
the complete extracted directory here with this exact name:

```text
dataset/
├── README.md                         # this file; tracked by Git
└── maizemask_v1.0_public_release/    # supplied data; ignored by Git
    ├── images/
    ├── metadata/
    │   └── images.csv
    ├── annotations/
    │   ├── maizemask_v1.0_coco.json
    │   ├── loso/
    │   ├── standard/
    │   └── random_tile_control/
    └── splits/
```

Do not rename the supplied release directory or move its `images/`,
`annotations/`, or `splits/` subdirectories. They are required by the training
data-preparation command:

Image names use `MM_D<field>_H<height>_S<source>_T<tile>.png`. Here, `D`, `H`,
and `S` are anonymized field, height class, and source-image aliases; `T` is a
deterministic within-source tile index. Use `metadata/images.csv` as the
machine-readable source of truth when creating a custom source-aware split.

```bash
python scripts/prepare_data_splits.py \
  --release-root dataset/maizemask_v1.0_public_release \
  --output-root .runtime_data
```

This creates loader-compatible LOSO, spatial-guard standard and random-tile
control trees. The paper runner consumes the LOSO tree by default:

```bash
python scripts/train_paper.py \
  --dataset-root .runtime_data/loso \
  --cache-root .cache \
  --output-root results/paper_run
```

The `.gitignore` rule excludes only
`dataset/maizemask_v1.0_public_release/`; never add the authorized data,
derived runtime data, raw captures, locations, checkpoints or results to Git.
