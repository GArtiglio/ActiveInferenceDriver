import torch
import torch.distributions as dist
from ebirl.distributions import NormalizingFlow, GaussianMixture, RealNVP
from ebirl.distributions import make_cov, get_mvn_dist

torch.manual_seed(123)

def test_make_cov():
    batch_size = 12
    dim = 3
    logvar = torch.randn(batch_size, dim)
    tril = torch.randn(batch_size, dim, dim)
    L1 = make_cov(logvar, tril, cholesky=True)
    
    # test var transform
    var = torch.exp(logvar)
    diag = torch.diagonal(L1, dim1=-2, dim2=-1)
    assert torch.all(torch.sum(var - diag, -1) == 0)
    
    print("test make cov passed")
    
def test_get_mvn_dist():
    batch_size = 12
    dim = 3
    offset = 0
    
    # test full cov
    params1 = torch.randn(batch_size, sum([dim, dim, dim**2]))
    dist1 = get_mvn_dist(params1, dim, "mvn", offset)
    mu1 = params1[:, :dim]
    var1 = torch.exp(params1[:, dim:dim*2])
    diag1 = torch.diagonal(dist1.scale_tril, dim1=-2, dim2=-1)
    tril1 = torch.tril(dist1.scale_tril, diagonal=-1)
    assert torch.all(torch.sum(dist1.loc - mu1, 1) == 0)
    assert torch.all(torch.sum(var1 - diag1, -1) == 0)
    assert torch.any(tril1 != 0)
    
    # test diag cov
    params2 = torch.randn(batch_size, sum([dim, dim]))
    dist2 = get_mvn_dist(params2, dim, "diag", offset)
    mu2 = params2[:, :dim]
    var2 = torch.exp(params2[:, dim:dim*2])
    diag2 = torch.diagonal(dist2.scale_tril, dim1=-2, dim2=-1)
    tril2 = torch.tril(dist2.scale_tril, diagonal=-1)
    assert torch.all(torch.sum(dist2.loc - mu2, 1) == 0)
    assert torch.all(torch.sum(var2 - diag2, -1) == 0)
    assert torch.all(tril2 == 0)
    
    print("test get mvn dist passed")

def test_norm_flow():
    dim = 3
    num_flows = 5
    flow = NormalizingFlow(dim, num_flows)
    
    # test single channel forward
    batch_size = 12
    x = torch.randn(batch_size, dim)
    y, log_det = flow(x)
    assert list(y.shape) == [batch_size, dim]
    assert list(log_det.shape) == [batch_size]
    
    # test multichannel forward
    channel_size = 10
    x = torch.randn(channel_size, batch_size, dim)
    y, log_det = flow(x)
    assert list(y.shape) == [channel_size, batch_size, dim]
    assert list(log_det.shape) == [channel_size, batch_size]
    
    print("test norm flow passed")

def test_gaussian_mixture():
    num_components = 3
    dim = 2
    cov = "mvn"
    batch_norm = True
    gmm = GaussianMixture(num_components, dim, cov, batch_norm)
    
    # test log probs
    batch_size = 12
    x = torch.randn(batch_size, dim)
    log_prob = gmm.log_prob(x)
    assert list(log_prob.shape) == [batch_size]
    
    # test sample
    num_samples = 32
    samples, logp = gmm.sample(num_samples)
    assert list(samples.shape) == [num_samples, dim]
    assert list(logp.shape) == [num_samples]
    
    print("test gauessian mixture passed")

def test_real_nvp():
    dim = 18
    hidden_dims = [32, 32]
    num_flows = 2
    mask = torch.zeros(num_flows, dim)
    mask[:, int(dim / 2):] = 1
    mask = torch.gather(mask, 1, torch.stack([torch.randperm(dim) for _ in range(num_flows)]).long())
    base = dist.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    seed = 0
    assert list(mask.shape) == [num_flows, dim]
    
    flow = RealNVP(dim, hidden_dims, mask, base, seed)
    
    # test single channel
    batch_size = 32
    z = torch.randn(batch_size, dim)
    x = flow(z)
    z_, log_det = flow.inverse(x)
    log_prob = flow.log_prob(x)
    samples, logp = flow.sample(batch_size)
    assert list(x.shape) == [batch_size, dim]
    assert list(z_.shape) == [batch_size, dim]
    assert list(log_det.shape) == [batch_size]
    assert list(log_prob.shape) == [batch_size]
    assert list(samples.shape) == [batch_size, dim]
    assert list(logp.shape) == [batch_size]
    
    print("test real nvp passed")

if __name__ == "__main__":
    test_make_cov()
    test_get_mvn_dist()
    test_norm_flow()
    test_gaussian_mixture()
    test_real_nvp()
    