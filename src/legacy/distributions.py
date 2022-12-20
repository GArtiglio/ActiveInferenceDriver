import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import pyro.distributions as pyro_dist
from .models import make_mlp

class NormalizingFlow(nn.Module):
    """ Implementation of Variational inference with normalizing flows, Rezende & Mohamed, 2015 
        This class chains a sequence of PlanarFlow modules, inverse is not supported
    """
    def __init__(self, dim, num_flows):
        super().__init__()
        self.flows = nn.ModuleList(
            [PlanarFlow(dim) for _ in range(num_flows)]
        )

    def sample(self, base_samples):
        """ transform samples from the based samples """
        samples = base_samples
        for flow in self.flows:
            samples = flow(samples)
        return samples

    def forward(self, x):
        """
        Computes and returns the sum of log_det_jacobians
        and the transformed samples T(x).
        """
        sum_log_det = 0
        transformed_sample = x
        for i in range(len(self.flows)):
            transformed_sample, log_det_i = self.flows[i](transformed_sample)
            sum_log_det += log_det_i
        return transformed_sample, sum_log_det
    
    
class PlanarFlow(nn.Module):
    """ Pyro implementation """
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.h_prime = lambda x: 1 - torch.tanh(x) ** 2
        self.init_params()
    
    def init_params(self):
        stdv = 1. / math.sqrt(self.u.shape[1])
        self.u.data.uniform_(-stdv, stdv)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-0.01, 0.01)
    
    def constrained_u(self):
        """ for invertibility """
        wu = self.u.matmul(self.w.T)
        m = lambda x: -1 + F.softplus(x)
        return self.u + (m(wu) - wu) * self.w.div(self.w.norm())

    def forward(self, z):
        u = self.constrained_u() 
        lin = z.matmul(self.w.T) + self.b
        x = z + u * self.h(lin)

        psi = self.h_prime(lin) * self.w
        log_det = torch.log(torch.abs(1 + psi.matmul(u.T))).squeeze(-1)
        return x, log_det


class GaussianMixture(nn.Module):
    def __init__(self, num_components, dim, cov="mvn", batch_norm=False, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        assert cov in ["diag", "mvn"]
        self.num_components = num_components
        self.dim = dim
        self.cov = cov
        self.batch_norm = batch_norm
        self.seed = seed

        self.pi_logits = nn.Parameter(torch.randn(num_components), requires_grad=True)
        self.mu = nn.Parameter(torch.randn(num_components, 1, dim), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(num_components, 1, dim), requires_grad=True)
        self.tril = nn.Parameter(torch.randn(num_components, 1, dim, dim), requires_grad=True)
        
        if batch_norm:
            self.bn = pyro_dist.transforms.BatchNorm(dim, momentum=0.1)
            self.bn.gamma.requires_grad = False
            self.bn.beta.requires_grad = False
            
        self.init_params()
    
    def __repr__(self):
        class_str = "{}(num_components={}, dim={}, cov={}, batch_norm={})".format(
            self.__class__.__name__,
            self.num_components,
            self.dim,
            self.cov,
            self.batch_norm
        )
        return class_str

    def init_params(self):
        if self.num_components == 1:
            nn.init.constant_(self.mu, 0)
            nn.init.constant_(self.logvar, 0)
            nn.init.constant_(self.tril, 0)
        else:
            nn.init.normal_(self.mu, 0, 1e-3)
            nn.init.normal_(self.logvar, 0, 1e-3)
            nn.init.normal_(self.tril, 0, 0.001)
    
    def get_obs_dist(self):
        if self.cov == "mvn":
            L = make_cov(self.logvar, self.tril, cholesky=True)
        else:
            L = torch.diag_embed(self.logvar.exp())
            
        P = dist.MultivariateNormal(self.mu, scale_tril=L)

        if self.batch_norm:
            P = pyro_dist.TransformedDistribution(P, [self.bn])
        return P

    def log_prob(self, x):
        # get dists
        pi = torch.softmax(self.pi_logits, dim=-1)
        P = self.get_obs_dist()

        logp_pi = torch.log(pi).unsqueeze(1)
        logp_P = P.log_prob(x)
        logp = torch.logsumexp(logp_pi + logp_P, dim=0)         
        return logp

    def sample(self, num_samples):
        # get dists
        pi = torch.softmax(self.pi_logits, dim=0).view(-1)
        P = self.get_obs_dist()

        # sample
        c = torch.multinomial(pi, num_samples, replacement=True)
        pi_samples = F.one_hot(c, num_classes=len(pi))
        P_samples = P.rsample((num_samples, )).squeeze(-2)

        samples = torch.sum(pi_samples.unsqueeze(-1) * P_samples, dim=1)
        logp = self.log_prob(samples)
        return samples, logp


class RealNVP(nn.Module):
    """ Implementation of Density estimation using real nvp, Dinh et al, 2016 """
    def __init__(self, dim, hidden_dims, mask, base, seed=0):
        """
        args:
            dim (int): dist dimension
            hidden_dims (int): hidden_dimensions
            mask (torch.tensor): autoregressive variable mask
            base (torch.dist): base distribution
            seed (int): default=0
        """
        super().__init__()
        torch.manual_seed(seed)
        self.base = base
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = nn.ModuleList([make_mlp(dim, dim, hidden_dims) for _ in range(len(mask))])
        self.s = nn.ModuleList([make_mlp(dim, dim, hidden_dims) for _ in range(len(mask))])
        
    def forward(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = torch.tanh(self.t[i](x_)) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x
    
    def inverse(self, x):
        z = x
        log_det_j = [torch.empty(0) for _ in range(len(self.t))]
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = torch.tanh(self.t[i](z_)) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_j[i] = s.sum(dim=-1)
        log_det_j = -torch.stack(log_det_j).sum(dim=0)
        return z, log_det_j
    
    def log_prob(self, x, base=None):
        base = self.base if base is None else base
        z, log_det_j = self.inverse(x)
        return base.log_prob(z) + log_det_j

    def sample(self, batch_size, base=None):
        base = self.base if base is None else base
        z = base.sample((batch_size,))
        x, log_det_j = self.inverse(z)

        logp = base.log_prob(z)
        logp -= log_det_j
        return x, logp


def make_cov(logvar, tril, cholesky=True):
    """ make full covarance matrix
    
    args:
        logvar (torch.tensor): log variance vector [batch_size, dim]
        tril (torch.tensor): unmaksed lower triangular matrix [batch_size, dim, dim]
        cholesky (bool): return cholesky decomposition, default=False
    
    returns:
        L (torch.tensor): scale_tril or cov [batch_size, dim, dim]
    """
    var = torch.exp(logvar.clip(math.log(1e-6), math.log(1e5)))
    L = torch.tril(tril, diagonal=-1)
    L = L + torch.diag_embed(var)
    
    if not cholesky:
        L = torch.bmm(L, L.transpose(-1, -2))
    return L

def get_mvn_dist(params, dim, dist_type, offset=0):
    """ get multivariate normal dist object
    
    args:
        params (torch.tensor): mvn param vector [batch_size, param_dim]
        dim (int): mvn dimension
        dist_type (str): if not "mvn" use diag cov
        offset ([float, torch.tensor]): logvar offset, default=0
        
    returns:
        mvn_dist (dist.MultivariateNormal)
    """
    if dist_type == "mvn":
        mu, logvar, tril = torch.split(params, [dim, dim, dim**2], dim=-1)
        tril = tril.view(-1, dim, dim)
    else:
        mu, logvar = torch.split(params, [dim, dim], dim=-1)
        tril = torch.zeros(len(params), dim, dim)

    logvar = logvar - offset
    L = make_cov(logvar, tril, cholesky=True)
    mvn_dist = dist.MultivariateNormal(mu, scale_tril=L)
    return mvn_dist

# def make_mlp(input_dim, output_dim, hidden_dims):
#     mlp = []
#     last_dim = input_dim
#     for h in hidden_dims:
#         mlp.append(nn.Linear(last_dim, h))
#         mlp.append(nn.LeakyReLU())
#         last_dim = h
#     mlp.append(nn.Linear(last_dim, output_dim))
#     return nn.Sequential(*mlp)