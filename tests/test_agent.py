import torch
import torch.nn as nn
from ebirl.agent import ActiveInference

torch.manual_seed(123)

def test_params_transform():
    state_dim = 4
    obs_dim = 3
    act_dim = 2
    horizon = 10
    precision = 1
    reward_ahead = True

    agent = ActiveInference(
        state_dim,
        obs_dim,
        act_dim,
        horizon,
        precision, 
        reward_ahead, 
    )
    
    batch_size = 32
    params = nn.Parameter(
        torch.randn(batch_size, sum(agent.num_params), requires_grad=True)
    )
    [A_mean, A_std, B, C, D, inv_tau, beta] = agent.transform_params(params)
    
    # test output shapes
    assert list(A_mean.shape) == [batch_size, obs_dim, state_dim]
    assert list(A_std.shape) == [batch_size, obs_dim, state_dim]
    assert list(B.shape) == [batch_size, act_dim, state_dim, state_dim]
    assert list(C.shape) == [batch_size, state_dim]
    assert list(D.shape) == [batch_size, state_dim]
    assert list(inv_tau.shape) == [batch_size, 1]
    assert list(beta.shape) == [batch_size, 1]
    
    # test output values
    assert torch.all(A_std > 0)
    assert torch.all(B.sum(-1) > 0)
    assert torch.all(C.sum(-1) > 0)
    assert torch.all(D.sum(-1) > 0)
    assert torch.all(inv_tau > 0)
    assert torch.all(beta > 0)
        
    print("test params transform passed")
    
def test_forward():
    state_dim = 4
    obs_dim = 3
    act_dim = 2
    horizon = 10
    precision = 1
    reward_ahead = True

    agent = ActiveInference(
        state_dim,
        obs_dim,
        act_dim,
        horizon,
        precision, 
        reward_ahead, 
    )
    
    batch_size = 32
    T = 12
    obs = torch.randn(batch_size, T, obs_dim).abs()
    act = torch.randint(act_dim, (batch_size, T))
    params = nn.Parameter(
        torch.randn(batch_size, sum(agent.num_params), requires_grad=True)
    )
    
    logp_a, logp_o = agent(params, obs, act)
    
    # test output shape
    assert list(logp_a.shape) == [batch_size, T]
    assert list(logp_o.shape) == [batch_size, T]
    
    # test output values
    assert torch.all(logp_a < 0)
    
    # test gradients
    loss = torch.mean(logp_a.sum(-1) + logp_o.sum(-1))
    loss.backward()
    assert torch.all(params.grad.data != 0)
    
    print("test forward passed")

if __name__ == "__main__":
    test_params_transform()
    test_forward()