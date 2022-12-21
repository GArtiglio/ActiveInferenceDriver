import math
import torch
import torch.nn as nn
import torch.distributions as torch_dist
import pyro.distributions as pyro_dist
import pyro.distributions.transforms as pyro_transform

from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from pyro.distributions.torch_transform import TransformModule

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


class SimpleTransformedModule(TransformedDistribution):
    """ Subclass of torch TransformedDistribution with mean, variance, entropy 
        implementation for simple transformations """
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args)
    
    @property
    def mean(self):
        mean = self.base_dist.mean
        for transform in self.transforms:
            if transform.__class__.__name__ == "BatchNormTransform":
                mean = transform._call(mean)
            elif transform.__class__.__name__ == "TanhTransform":
                mean = transform._call(mean)
            else:
                raise NotImplementedError
        return mean
    
    @property
    def variance(self):
        variance = self.base_dist.variance
        for transform in self.transforms:
            if transform.__class__.__name__ == "BatchNormTransform":
                variance *= transform.moving_variance / transform.constrained_gamma**2
            else:
                raise NotImplementedError
        return variance
    
    @property
    def covariance_matrix(self):
        if self.base_dist.__class__.__name__ != "MultivariateNormal":
            raise NotImplementedError
        else:
            covariance_matrix = self.base_dist.covariance_matrix
            for transform in self.transforms:
                if transform.__class__.__name__ == "BatchNormTransform":
                    w = transform.moving_variance**0.5 / transform.constrained_gamma
                    w_mask = w.unsqueeze(-1) * w.unsqueeze(-2)
                    covariance_matrix *= w_mask
                else:
                    raise NotImplementedError
        return covariance_matrix

    def entropy(self):
        entropy = self.base_dist.entropy()
        for transform in self.transforms:
            if transform.__class__.__name__ == "BatchNormTransform":
                scale = torch.sqrt(transform.moving_variance) / transform.constrained_gamma
                entropy += torch.log(scale).sum()
            elif transform.__class__.__name__ == "TanhTransform": # skip tanh transform
                pass
            else:
                pass
        return entropy


class BatchNormTransform(TransformModule):
    """ Masked batchnorm transform adapted from pyro's implementation. 
    Masks are inferred from observations assuming unmasked observations will not be exactly zero.
    Otherwise we treat them as zero padded.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0
    def __init__(self, input_dim, momentum=0.1, epsilon=1e-5, affine=False, update_stats=True):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.update_stats = update_stats
        self.epsilon = epsilon
         
        self.moving_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.moving_variance = nn.Parameter(torch.ones(input_dim), requires_grad=False)
        
        self.gamma = nn.Parameter(torch.ones(input_dim), requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(input_dim), requires_grad=affine)
            
    @property   
    def constrained_gamma(self):
        """ Enforce positivity """
        return torch.relu(self.gamma) + 1e-6
    
    def _call(self, x):
        return (x - self.beta) / self.constrained_gamma * torch.sqrt(
            self.moving_variance + self.epsilon
        ) + self.moving_mean
            
    def _inverse(self, y):
        op_dims = [i for i in range(len(y.shape) - 1)]
        
        if self.training and self.update_stats:
            mask = 1. - 1. * torch.all(y == 0, dim=-1, keepdim=True)
            mean = torch.sum(mask * y, dim=op_dims) / (mask.sum(op_dims) + 1e-6)
            var = torch.sum(mask * (y - mean)**2, dim=op_dims) / (mask.sum(op_dims) + 1e-6)
            with torch.no_grad():
                self.moving_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.moving_variance.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            mean, var = self.moving_mean, self.moving_variance
        
        return (y - mean) * self.constrained_gamma / torch.sqrt(
            var + self.epsilon
        ) + self.beta
    
    def log_abs_det_jacobian(self, x, y):
        op_dims = [i for i in range(len(y.shape) - 1)]
        if self.training and self.update_stats:
            mask = 1. - 1. * torch.all(y == 0, dim=-1, keepdim=True)
            mean = torch.sum(mask * y, dim=op_dims, keepdim=True) / (mask.sum(op_dims) + 1e-6)
            var = torch.sum(mask * (y - mean)**2, dim=op_dims, keepdim=True) / (mask.sum(op_dims) + 1e-6)
        else:
            var = self.moving_variance
        return -self.constrained_gamma.log() + 0.5 * torch.log(var + self.epsilon)