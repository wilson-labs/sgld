from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from .metrics import ece


def set_seeds(seed=None):
  if seed is not None and seed >= 0:
    import random
    random.seed(seed)

    try:
      import numpy as np
      np.random.seed(seed)
    except ImportError:
      pass

    try:
      import torch
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
    except ImportError:
      pass


@torch.no_grad()
def test_sample(data_loader, net, likelihood, device=None, return_logits_targets=False):
  net.eval()

  N = len(data_loader.dataset)

  nll = 0.
  N_acc = 0
  all_logits, all_targets = [], []

  for X, Y in tqdm(data_loader, leave=False):
    X, Y = X.to(device), Y.to(device)
    
    logits = net(X)

    nll += likelihood(logits, Y)
    N_acc += (logits.argmax(dim=-1) == Y).sum()
    all_logits.append(logits)
    all_targets.append(Y)

  metrics = { 'nll': nll.item(), 'acc': (N_acc / N).item() }
  if return_logits_targets:
    metrics['logits'] = torch.cat(all_logits)
    metrics['targets'] = torch.cat(all_targets)
  
  return metrics


@torch.no_grad()
def test_bma(samples_dir, data_loader, net, likelihood, device=None):
  net.eval()

  bma_logits = []
  bma_nll = []
  all_targets = None

  for sample_path in tqdm(Path(samples_dir).rglob(f'*.sample.pt'), leave=False):
    net.load_state_dict(torch.load(sample_path))

    metrics = test_sample(data_loader, net, likelihood,
                          device=device, return_logits_targets=True)
    
    bma_logits.append(metrics['logits'])
    bma_nll.append(metrics['nll'])
    all_targets = metrics['targets']

  bma_nll = torch.Tensor(bma_nll).mean(dim=-1)
  bma_probs = torch.stack(bma_logits).softmax(dim=-1).mean(dim=0)
  bma_ece, _ = ece(
    F.one_hot(all_targets, bma_probs.size(-1)).cpu().numpy(),
    bma_probs.cpu().numpy(), num_bins=30)
  bma_acc = (bma_probs.argmax(dim=-1) == all_targets).float().mean()

  return { 'bma_nll': bma_nll, 'bma_acc': bma_acc, 'bma_ece': bma_ece }
