import math
import torch
import torch.nn as nn
import torch.distributions as torch_dist
import pyro.distributions as pyro_dist
import pyro.distributions.transforms as pyro_transform

def rectify(x, low=1e-6, high=1e6):
    low, high = math.log(low), math.log(high)
    out = torch.exp(x.clip(low, high))
    return out

def make_cov(lv, tl, cholesky=True):
    """ Make full covarance matrix
    
    args:
        lv (torch.tensor): log variance vector [batch_size, dim]
        tl (torch.tensor): unmaksed lower triangular matrix [batch_size, dim, dim]
        cholesky (bool): return cholesky decomposition, default=False
    
    returns:
        L (torch.tensor): scale_tril or cov [batch_size, dim, dim]
    """
    variance = rectify(lv)
    L = torch.tril(tl, diagonal=-1)
    L = L + torch.diag_embed(variance)
    
    if not cholesky:
        L = torch.bmm(L, L.transpose(-1, -2))
    return L

def make_strong_diag_matrix(x):
    """ Make diagonally dominant matrix """
    row_sum = x.abs().sum(-1)
    x_ = x + torch.diag_embed(row_sum)
    return x_

def kl_divergence(p, q, eps=1e-6):
    """ Discrete kl divergence """
    assert p.shape[-1] == q.shape[-1]
    log_p = torch.log(p + eps)
    log_q = torch.log(q + eps)
    kl = torch.sum(p * (log_p - log_q), dim=-1)
    return kl
    
def poisson_pdf(gamma, K):
    """ 
    Args:
        gamma (torch.tensor): poission arrival rate [batch_size, 1]
        K (int): number of bins

    Returns:
        pdf (torch.tensor): truncated poisson pdf [batch_size, K]
    """
    assert torch.all(gamma > 0)
    Ks = 1 + torch.arange(K).to(gamma.device)
    poisson_dist = torch.distributions.Poisson(gamma)
    pdf = torch.softmax(poisson_dist.log_prob(Ks), dim=-1)
    return pdf

class FlowMVN(nn.Module):
    def __init__(self, dim, cov="full", lv0=0., use_bn=True):
        """ Multivariate normal with batchnorm flow
        Args:
            dim
            cov (str, optional)
            lv0 (float, optional): initial log variance. Default=0.
            use_bn (bool, optional)
        """
        super().__init__()
        self.dim = dim
        self.cov = cov
        self.use_bn = use_bn

        self.mu = nn.Parameter(0.1 * torch.randn(1, dim))
        self.lv = nn.Parameter(lv0 * torch.ones(1, dim))

        # nn.init.xavier_normal_(self.mu)
        
        self.tl = torch.zeros(1, dim, dim)
        if cov == "full":
            self.tl = nn.Parameter(self.tl)
        
        if self.use_bn:
            self.bn = pyro_transform.BatchNorm(dim, momentum=0.1)
            self.bn.gamma.requires_grad = False
            self.bn.beta.requires_grad = False
    
    def __repr__(self):
        s = "{}(dim={}, cov={}, use_bn={})".format(
            self.__class__.__name__, self.dim, self.cov, self.use_bn
        )
        return s

    def get_distribution_class(self):
        L = make_cov(self.lv, self.tl)
        dist = torch_dist.MultivariateNormal(self.mu, scale_tril=L)
        if self.use_bn:
            dist = pyro_dist.TransformedDistribution(dist, [self.bn])
        return dist

    def log_prob(self, x):
        logp = self.get_distribution_class().log_prob(x)
        return logp