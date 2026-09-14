# Authorized MaizeMask v1.0 data

This directory is tracked so the expected dataset location is visible after a
GitHub clone. The dataset itself is not in this repository.

After receiving an authorized MaizeMask v1.0 release from the authors, place
the complete extracted directory here with this exact name:

```text
dataset/
├── README.md                         # this file; tracked by Git
└── maizemask_v1.0_public_release/    # supplied data; ignored by Git
    ├── images/
    ├── annotations/
    │   ├── maizemask_v1.0_coco.json
    │   ├── loso/
    │   └── standard/
    └── splits/
```

Do not rename the supplied release directory or move its `images/`,
`annotations/`, or `splits/` subdirectories. They are required by the training
data-preparation command:

```bash
python scripts/prepare_data_splits.py \
  --release-root dataset/maizemask_v1.0_public_release \
  --output-root .runtime_data
```

This creates the loader-compatible LOSO tree consumed by the paper runner:

```bash
python scripts/train_paper.py \
  --dataset-root .runtime_data/loso \
  --cache-root .cache \
  --output-root results/paper_run
```

The `.gitignore` rule excludes only
`dataset/maizemask_v1.0_public_release/`; never add the authorized data,
derived runtime data, raw captures, locations, checkpoints or results to Git.
