import logging
from pathlib import Path
import random
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from sgld.models import ResNet18
from sgld.optim import SGLD
from sgld.optim.lr_scheduler import CosineLR


def set_seeds(seed=None):
  if seed is not None and seed >= 0:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_test_split(dataset, val_size=.1, seed=None):
    N = len(dataset)
    N_test = int(val_size * N)
    N -= N_test

    if seed is not None:
        train, test = random_split(dataset, [N, N_test], 
                                   generator=torch.Generator().manual_seed(seed))
    else:
        train, test = random_split(dataset, [N, N_test])

    return train, test


def get_cifar10(root=None, val_size=.1, seed=42, augment=True):
  _CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  _CIFAR_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  train_data = CIFAR10(root=root, train=True, download=True,
                       transform=_CIFAR_TRAIN_TRANSFORM if augment else _CIFAR_TEST_TRANSFORM)

  test_data = CIFAR10(root=root, train=False, download=True,
                      transform=_CIFAR_TEST_TRANSFORM)

  if val_size != 0:
    train_data, val_data = train_test_split(train_data, val_size=val_size, seed=seed)
    return train_data, val_data, test_data

  return train_data, test_data


@torch.no_grad()
def test(data_loader, net, criterion, device=None):
  net.eval()

  total_loss = 0.
  N = 0
  Nc = 0

  for X, Y in tqdm(data_loader, leave=False):
    X, Y = X.to(device), Y.to(device)
    
    f_hat = net(X)
    Y_pred = f_hat.argmax(dim=-1)
    loss = criterion(f_hat, Y)

    N += Y.size(0)
    Nc += (Y_pred == Y).sum().item()
    total_loss += loss

  avg_loss = total_loss.item() / N
  acc = Nc / N

  return {
    'avg_loss': avg_loss,
    'acc': acc,
  }


@torch.no_grad()
def test_bma(net, data_loader, samples_dir, device=None):
  net.eval()

  ens_logits = []

  for sample_path in tqdm(Path(samples_dir).rglob('*.pt')):
    net.load_state_dict(torch.load(sample_path))

    all_logits = []
    all_Y = []
    for X, Y in tqdm(data_loader, leave=False):
      X, Y = X.to(device), Y.to(device)
      all_logits.append(net(X))
      all_Y.append(Y)
    all_logits = torch.cat(all_logits)
    all_Y = torch.cat(all_Y)

    ens_logits.append(all_logits)

  ens_logits = torch.stack(ens_logits).softmax(dim=-1).mean(dim=0)
  Y_pred = ens_logits.argmax(dim=-1)

  acc = (Y_pred == all_Y).sum().item() / Y_pred.size(0)

  return { 'acc': acc }  


def main(seed=None, device=0, data_dir=None, val_size=.1, aug=True, epochs=1,
         batch_size=128, lr=.5, momentum=.9, weight_decay=5e-4, n_cycles=4,
         temperature=1/50000, n_samples=12, samples_dir=None):
  if data_dir is None and os.environ.get('DATADIR') is not None:
      data_dir = os.environ.get('DATADIR')

  samples_dir = Path(samples_dir or '.') / '.samples'
  samples_dir.mkdir()

  torch.backends.cudnn.benchmark = True

  set_seeds(seed)
  device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

  train_data, val_data, test_data = get_cifar10(root=data_dir, val_size=val_size,
                                                seed=seed, augment=bool(aug))
  
  train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2,
                            shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2)
  test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

  net = ResNet18(num_classes=10).to(device)
  criterion = nn.CrossEntropyLoss()

  sgld = SGLD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum,
              temperature=temperature)
  sgld_scheduler = CosineLR(sgld, n_cycles=n_cycles, n_samples=n_samples,
                            T_max=len(train_loader) * epochs)

  for e in tqdm(range(epochs)):
    net.train()
    for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
      X, Y = X.to(device), Y.to(device)

      sgld.zero_grad()

      f_hat = net(X)
      loss = criterion(f_hat, Y)
      
      loss.backward()

      if sgld_scheduler.get_last_beta() < sgld_scheduler.beta:
        sgld.step(noise=False)
      else:
        sgld.step()

        if sgld_scheduler.should_sample():
          torch.save(net.state_dict(), samples_dir / f's_e{e}_m{i}.pt')

      sgld_scheduler.step()

    ## NOTE: Only for sanity checks, not BMA.
    val_metrics = test(val_loader, net, criterion, device=device)
    test_metrics = test(test_loader, net, criterion, device=device)

    logging.info(f"(Epoch {e}) Val/Test: {val_metrics['acc']:.4f} / {test_metrics['acc']:.4f}")

  bma_val_metrics = test_bma(net, val_loader, samples_dir, device=device)
  bma_test_metrics = test_bma(net, test_loader, samples_dir, device=device)

  logging.info(f"BMA Val/Test: {bma_val_metrics['acc']:.4f} / {bma_test_metrics['acc']:.4f}")


if __name__ == '__main__':
  import fire
  import os

  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(main)
