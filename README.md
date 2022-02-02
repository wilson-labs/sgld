# SGLD

An implementation of SGLD (and variants) with two key purposes in mind:
1. Easy to cherry-pick into your own *research* code for baseline evaluation.
2. Clean enough for dirty prototyping with fancy/ad-hoc extensions.

## Setup

Use the [environment.yml](./environment.yml) to add common dependencies to
your working Python environment.

```shell
conda env create -n <env_name>
```

Add the `slgd` package to PYTHONPATH for imports.

```shell
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## Usage

### Toy SGLD Visualization

See [viz_sgld.ipynb](./notebooks/viz_sgld.ipynb).

### Full Example

Full examples which are easy to modify.

- **SGLD**: A full pipeline of using SGLD is provided in [train_sgld.py](./experiments/train_sgld.py)

- **cSGLD**: A full pipeline of using SGLD + cyclical step size schedule (cSGLD) is provided in [train_csgld.py](./experiments/train_csgld.py)

## License

MIT
