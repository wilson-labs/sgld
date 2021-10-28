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
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## Usage

### Toy Example

See [viz_sgld.ipynb](./notebooks/viz_sgld.ipynb).

### CIFAR-10

See [train_cifar10.py](./experiments/train_cifar10_csgld.py) which using cSGHMC to
train on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

#### No Data Augmentation

The default arguments should be enough to achieve ~85% with no tempering and
no data augmentation.

```shell
python experiments/train_cifar10_csgld.py --epochs=1000
```

#### With Data Augmentation

Use the following additional CLI arguments to achieve ~94% with data augmentation
and tempering.

```shell
python experiments/train_cifar10_csgld.py --epochs=1000 \
                                          --aug=1 \
                                          --momentum=0.95 \
                                          --temperature=2e-5
```

## License

MIT
