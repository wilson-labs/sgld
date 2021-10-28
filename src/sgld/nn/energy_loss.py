import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianPriorCELoss(nn.Module):
  '''Scaled CrossEntropy + Gaussian prior

  To get the unbiased density estimate, multiply by N.
  '''
  def __init__(self, params, prior_scale=1):
    super().__init__()

    self.theta = params
    self.sigma = prior_scale

    self.ce = nn.CrossEntropyLoss()

  def forward(self, logits, Y, N=1):
    energy = self.ce(logits, Y)
    
    for p in self.theta:
      prior = Normal(torch.zeros_like(p), self.sigma)
      energy -= prior.log_prob(p).sum() / N
    
    return energy
