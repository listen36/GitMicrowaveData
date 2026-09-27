# Contributing

Please open an issue before proposing a change that alters the cohort, split, feature definition, hyperparameter search, or statistical procedure. Such changes create a new experiment identity and must not silently overwrite the reported results.

For code changes:

1. Create a focused branch.
2. Install the development environment with `python -m pip install -e ".[dev]"`.
3. Run `pytest`.
4. State whether results were regenerated, and list the data release, split, seed, and command.
5. Do not commit UM-BMID signal files, local paths, credentials, virtual environments, or logs.

Any added evaluation must keep samples from the same physical phantom group out of both development and evaluation partitions. If a threshold is reported, document exactly where it was selected.

