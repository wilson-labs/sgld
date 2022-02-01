import torch
import torch.nn as nn


class SoftmaxNLL(nn.Module):
    '''
    -ve log (tempered or logit-scaled) softmax likelihood.
    '''
    def __init__(self, temperature=1, logits_temperature=1):
        super().__init__()

        self.T = temperature
        self.logits_T = logits_temperature

        self.nll = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, logits, Y):
        return self.nll(logits / self.logits_T, Y).div(self.T)


class GaussianNL(nn.Module):
    '''
    -ve log independent Gaussian prior over parameters.
    '''
    def __init__(self, prior_scale=1):
        super().__init__()

        self.sigma = prior_scale

    def forward(self, params):
        neg_log_prob = 0
        for p in params:
            prior = torch.distributions.Normal(torch.zeros_like(p), self.sigma)
            neg_log_prob -= prior.log_prob(p).sum()
        return neg_log_prob
