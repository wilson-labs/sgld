# SGLD

This is a standalone and easy-to-extend implementation of SGLD (plus a few
variants).

## Setup

Use the [requirements.txt](./requirements.txt) to add common dependencies to
your working Python environment.

```shell
pip install -r requirements.txt
```

Add the `slgd` package to PYTHONPATH as

```shell
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Usage

For a bare bones example on a toy density, see [viz_sgld.ipynb](./notebooks/viz_sgld.ipynb).

For a non-trivial example, see [train_cifar10.py](./experiments/train_cifar10.py). This
should be able to achieve 95%+ test accuracy with the default arguments when trained
for 200 epochs. The secret sauce consists of cSGLD + temperature scaling.

## License

MIT
