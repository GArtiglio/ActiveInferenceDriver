import torch
import torch.nn as nn
import torch.distributions as torch_dist
from src.agent import ActiveInference
from src.distributions import FlowMVN
from src.distributions import rectify

class EBIRL(nn.Module):
    """ Empirical Bayes inverse reinforcement learning """
    def __init__(self, state_dim, act_dim, horizon, 
        obs_dist="norm", a_cum=False, prior_cov="full", obs_penalty=1., prior_penalty=1.):
        """
        Args:
            state_dim
            act_dim
            horizon
            obs_dist
            a_cum
            prior_cov (str, optional): prior covariance type. Default=full
            obs_penalty (float): prior obs penalty. Default=1.
            prior_penalty (float): prior likelihood penalty. Default=1.
        """
        super().__init__()
        self.obs_penalty = obs_penalty
        self.prior_penalty = prior_penalty

        self.agent = ActiveInference(
            state_dim, act_dim, horizon, obs_dist=obs_dist, a_cum=a_cum
        )
        num_params = sum(self.agent.num_params)
        self.prior = FlowMVN(
            num_params, cov=prior_cov, lv0=-2., use_bn=False
        )
        self.plot_keys = ["total_loss", "act_loss", "obs_loss", "prior_logp", "prior_ent"]
        
    def get_stdout(self, stats):
        s = "total_loss: {:.4f}, act_loss: {:.4f}, obs_loss: {:.4f}, prior_logp: {:.4f}, prior_ent: {:.4f}".format(
            stats['total_loss'], stats['act_loss'], stats['obs_loss'], stats["prior_logp"], stats["prior_ent"]
        )
        return s
    
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
        _, act_stats = self.agent.compute_act_loss(o, a, out, mask)
        obs_loss, obs_stats = self.agent.compute_obs_loss(o, a, out, mask)
        
        loss = self.obs_penalty * obs_loss.mean()
        
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
            "total_loss": loss.data.item(),
            **act_stats, **obs_stats,
            "prior_logp": prior_logp.data.mean().item(),
            "prior_ent": q_ent.data.mean().item()
        }
        return loss, stats
    
    def compute_loss(self, o, a, mask):
        prior_loss, prior_stats = self.compute_prior_loss(o, a, mask)
        post_loss, post_stats = self.compute_posterior_loss(o, a, mask)
        loss = prior_loss + post_loss

        prior_stats = {f"prior_{k}":v for (k, v) in prior_stats.items()}
        stats = {
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
        state_dim, act_dim, horizon, obs_dist="norm", prior_cov="full", 
        obs_penalty=obs_penalty, prior_penalty=prior_penalty
    )
    model.init_q(batch_size, freeze_prior=False)
    print(model)
    loss, stats = model.compute_marginal_loss(o, a, mask)
    loss.backward()
    
    # grad check
    print("grads")
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n, None)
        else:
            print(n, p.grad.data.norm())
