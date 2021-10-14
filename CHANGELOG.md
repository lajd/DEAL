# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2021-10-09

### Changes
- Refactor code, supporting only inductive setting
- Removed unused methods/modules
- Add install instructions to README
- Provide results/reproduction steps in README
- Update to latest version of torch geometric (2.0.1)
  - Use datasets and utilities from torch geometric
- Refactor datasets creation
  - ** Note: Datasets are not the same as in the original paper
      - Graphs are undirected (twice the number of edges)
      - Different train/validation/test splits
      - Different method for negative sampling, negative sampling fraction
- Support both inductive and transductive validation
  - For transductive validation, the Validation set subgraph is part of training set. For inductive validation, these nodes/edges
  are unseen during training.
  - All inference on the test set is inductive
    - We set lambda_0 -> 0 so that only node attributes are taken into effect
