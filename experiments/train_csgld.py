import logging
import os
from pathlib import Path
from uuid import uuid4
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

import sgld.nn as snn
from sgld.optim import SGLD
from sgld.optim.lr_scheduler import CosineLR
from sgld.train_utils import set_seeds, test_sample, test_bma
from sgld.dataset import get_cifar10


def run_csgld(train_loader, test_loader, net, likelihood, prior, samples_dir,
              device=None, lr=1e-6, momentum=.9, temperature=1,
              n_samples=20, n_cycles=1, epochs=1):
  N = len(train_loader.dataset)

  sgld = SGLD(net.parameters(), lr=lr, momentum=momentum, temperature=temperature)
  sgld_scheduler = CosineLR(sgld, n_cycles=n_cycles, n_samples=n_samples,
                            T_max=len(train_loader) * epochs)

  for e in tqdm(range(epochs)):
    net.train()
    for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
      X, Y = X.to(device), Y.to(device)
      M = len(Y)

      sgld.zero_grad()

      nll = likelihood(net(X), Y).mul(N / M)
      nlp = prior(net.parameters())
      loss = nll + nlp

      loss.backward()

      if sgld_scheduler.get_last_beta() < sgld_scheduler.beta:
        sgld.step(noise=False)
      else:
        sgld.step()

        if sgld_scheduler.should_sample():
          torch.save(net.state_dict(), samples_dir / f'e{e}_m{i}.sample.pt')

          bma_test_metrics = test_bma(samples_dir, test_loader, net, likelihood, device=device)
          logging.info({f'csgld/test/{k}': v for k, v in bma_test_metrics.items() })

      sgld_scheduler.step()

      if i % 100 == 0:
        metrics = {'epoch': e, 'minibatch': i,
                   'nll': nll.item(), 'nlp': nlp.item(), 'loss': loss.item()}
        logging.debug({f'csgld/train/{k}': v for k, v in metrics.items() })

    test_metrics = test_sample(test_loader, net, likelihood, device=device)
    logging.info({f'csgld/test/{k}': v for k, v in test_metrics.items() })

  bma_test_metrics = test_bma(samples_dir, test_loader, net, likelihood, device=device)
  logging.info({f'csgld/test/{k}': v for k, v in bma_test_metrics.items() })


def main(seed=None, device=0, data_dir=None, augment=True, batch_size=128,
         ckpt_path=None, prior_scale=1, temperature=1,
         epochs=1000, lr=1e-6, momentum=.9, n_cycles=50, n_samples=50, samples_dir=None):
  set_seeds(seed)
  device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"
  torch.backends.cudnn.benchmark = True

  if data_dir is None and os.environ.get('DATADIR') is not None:
    data_dir = os.environ.get('DATADIR')
    logging.warning(f'Using default data directory from environment "{data_dir}".')

  samples_dir = Path(samples_dir or f'./.log/csgld-{str(uuid4())[:8]}') / 'samples'
  samples_dir.mkdir(parents=True)
  logging.info(f'Storing samples in "{samples_dir.resolve()}".')

  logging.info({
    'seed': seed,
    'device': device,
    'augment': augment,
    'batch_size': batch_size,
    'prior_scale': prior_scale,
    'temperature': temperature,
    'lr': lr,
    'momentum': momentum,
    'n_cycles': n_cycles,
    'n_samples': n_samples,
  })

  train_data, test_data = get_cifar10(root=data_dir, augment=bool(augment))

  train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2,
                            shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

  net = snn.ResNet18(num_classes=10).to(device)
  if ckpt_path is not None and Path(ckpt_path).resolve().is_file():
    net.load_state_dict(torch.load(ckpt_path))
    logging.info(f'Loaded checkpoint "{ckpt_path}".')

  likelihood = snn.SoftmaxNLL()
  prior = snn.GaussianNL(prior_scale=prior_scale)

  run_csgld(train_loader, test_loader, net, likelihood, prior, samples_dir,
            device=device, lr=lr, momentum=momentum, temperature=temperature,
            n_cycles=n_cycles, n_samples=n_samples, epochs=epochs)


if __name__ == '__main__':
  import fire
  import os
  import sys

  logging.basicConfig(
    stream=sys.stdout, format='[%(asctime)s] %(levelname)s: %(message)s',
    level=getattr(logging, os.environ.get('LOGLEVEL', 'INFO'), logging.INFO))
  
  fire.Fire(main)
