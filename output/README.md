# Output Files

Training logs and model checkpoints are saved in this directory.

The training script creates one subdirectory for each split key, for example:

```text
output/
└── Drug_unseen/
    ├── best_model_<timestamp>.pth
    └── train_<timestamp>.log
```

Generated checkpoints and logs are not included in this repository.
