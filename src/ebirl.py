import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agent import ActiveInference
from src.distributions import FlowMVN
from src.distributions import rectify

class EBIRL(nn.Module):
    """ Empirical Bayes inverse reinforcement learning """
    def __init__(self, state_dim, act_dim, horizon, prior_cov="full", 
        bc_penalty=1., obs_penalty=1., prior_penalty=1.):
        """
        Args:
            state_dim (int): agent hidden state dimension
            act_dim (int): agent action dimension
            horizon (int): agent max planning horizon
            prior_cov (str, optional): prior covariance type. Choices=["full", "diag"]
            bc_penalty (float): prior bc penalty. Default=1.
            obs_penalty (float): prior obs penalty. Default=1.
            prior_penalty (float): prior likelihood penalty. Default=1.
        """
        super().__init__()
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty
        self.prior_penalty = prior_penalty

        self.agent = ActiveInference(
            state_dim, act_dim, horizon
        )
        num_params = sum(self.agent.num_params)
        self.prior = FlowMVN(
            num_params, cov=prior_cov, lv0=-2., use_bn=False
        )
        self.plot_keys = ["total_loss", "act_loss", "obs_loss", "prior_logp", "post_ent"]

        self.init_params()
        
    def get_stdout(self, stats):
        s = "total_loss: {:.4f}, act_loss: {:.4f}, obs_loss: {:.4f}, prior_logp: {:.4f}, post_ent: {:.4f}".format(
            stats['total_loss'], stats['act_loss'], stats['obs_loss'], stats["prior_logp"], stats["post_ent"]
        )
        return s
    
    def init_params(self):
        """ sparse and identity initialization """
        state_dim = self.agent.state_dim
        act_dim = self.agent.act_dim

        A_mean = torch.linspace(-1, 1, state_dim)
        A_log_std = torch.log(torch.ones(state_dim) / state_dim)
        B = torch.eye(state_dim).unsqueeze(0).repeat_interleave(act_dim, 0).flatten()
        C = torch.zeros(state_dim)
        D = torch.zeros(state_dim)
        tau = torch.zeros(1)
        
        params = torch.cat([A_mean, A_log_std, B, C, D, tau], dim=-1).unsqueeze(0)
        self.prior.mu.data = params

    def init_q(self, batch_size, freeze_prior=False):
        """ Initialize variational distributions """
        prior_mu = self.prior.mu.data.clone()
        prior_lv = self.prior.lv.data.clone()
        prior_mu = torch.repeat_interleave(prior_mu, batch_size, dim=0)
        prior_lv = torch.repeat_interleave(prior_lv, batch_size, dim=0)
        self.q_mu = nn.Parameter(prior_mu)
        self.q_lv = nn.Parameter(prior_lv)

        if freeze_prior:
            for p in self.prior.parameters():
                p.requires_grad = False

    def get_q_dist(self):
        dist = torch_dist.Normal(self.q_mu, rectify(self.q_lv))
        return dist

    def compute_prior_loss(self, o, a, mask):
        prior_dist = self.prior.get_distribution_class()
        theta_prior = prior_dist.rsample()
        theta_prior = torch.repeat_interleave(theta_prior, o.shape[1], dim=0)

        out = self.agent.forward(o, a, theta_prior, detach=True)
        act_loss, act_stats = self.agent.compute_act_loss(o, a, out, mask)
        obs_loss, obs_stats = self.agent.compute_obs_loss(o, a, out, mask)
        
        loss = self.bc_penalty * act_loss.mean() + self.obs_penalty * obs_loss.mean()
        
        stats = {
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats
        }
        return loss, stats

    def compute_posterior_loss(self, o, a, mask):
        q_dist = self.get_q_dist()
        theta = q_dist.rsample()

        out = self.agent.forward(o, a, theta, detach=False)
        act_loss, act_stats = self.agent.compute_act_loss(o, a, out, mask)
        _, obs_stats = self.agent.compute_obs_loss(o, a, out, mask)
        
        prior_logp = self.prior.log_prob(theta)
        q_ent = q_dist.entropy().sum(-1)
        loss = torch.mean(act_loss + self.prior_penalty * (-prior_logp - q_ent))
        
        stats = {
            "post_total_loss": loss.data.item(),
            **act_stats, **obs_stats,
            "prior_logp": prior_logp.data.mean().item(),
            "post_ent": q_ent.data.mean().item()
        }
        return loss, stats
    
    def compute_loss(self, o, a, mask):
        prior_loss, prior_stats = self.compute_prior_loss(o, a, mask)
        post_loss, post_stats = self.compute_posterior_loss(o, a, mask)
        loss = prior_loss + post_loss

        prior_stats = {f"prior_{k}":v for (k, v) in prior_stats.items()}
        stats = {
            "total_loss": loss.data.item(),
            **post_stats, **prior_stats
        }
        return loss, stats

if __name__ == "__main__":
    torch.manual_seed(0)
    state_dim = 10
    act_dim = 3
    horizon = 30
    obs_penalty = 1.
    prior_penalty = 1.
    
    # synthetic data
    T = 24
    batch_size = 64
    o = torch.randn(T, batch_size).abs()
    a = torch.randint(0, 2, (T, batch_size))
    mask = torch.randint(0, 2, (T, batch_size))

    model = EBIRL(
        state_dim, act_dim, horizon,
        obs_penalty=obs_penalty, prior_penalty=prior_penalty
    )
    model.init_q(batch_size, freeze_prior=False)
    print(model)
    loss, stats = model.compute_loss(o, a, mask)
    loss.backward()
    
    # grad check
    print("grads")
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n, None)
        else:
            print(n, p.grad.data.norm())
