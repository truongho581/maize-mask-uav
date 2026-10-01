# MaizeMask v1.0 dataset

This directory is tracked so the expected dataset location is visible after a
GitHub clone. The dataset itself is archived on Zenodo rather than duplicated
in this repository.

## Dataset download

Download the canonical `MaizeMask_v1.0.zip` archive from:

- Zenodo record: <https://zenodo.org/records/23058644>
- Version DOI: <https://doi.org/10.5281/zenodo.23058644>

The published archive is version 1.0.0 and is licensed under CC BY 4.0.

After downloading, place the complete extracted directory here with this exact
name:

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
`dataset/maizemask_v1.0_public_release/`; never add the downloaded data,
derived runtime data, raw captures, locations, checkpoints or results to Git.
