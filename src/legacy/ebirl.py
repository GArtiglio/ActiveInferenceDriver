import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent import ActiveInference
from .models import RNN, KLScheduler
from .distributions import NormalizingFlow, GaussianMixture, get_mvn_dist

class Encoder(nn.Module):
    """ Variational distribution of agent parameters
    
    Iterative inference: the model stores parameter distributions 
        for all trajectories in the dataset
    Amortized inference: the model uses an rnn to output trajectory parameter distributions.
    """
    def __init__(
        self, 
        input_dim, 
        z_dim, 
        hidden_dim, 
        stack_dim, 
        dist_type, 
        num_flows=5, 
        iterative=False, 
        batch_size=32, 
        init_uniform=True, 
        logvar_offset=0, 
        dropout=0, 
        bidirectional=False, 
        seed=0
        ):
        """
        args:
            input_dim (int): agent observation dimension
            z_dim (int): number of agent parameters
            hidden_dim (int): rnn hidden size
            stack_dim (int): rnn layer size
            dist_type (str): agent parameters dist type ["diag", "mvn", "norm_flow"]
            num_flows (int, optional): number of flow layers, default=5.
            iterative (bool, optional): whether use iterative inference, default=False.
            batch_size (int, optional): batch size required for iterative inference, default=32.
            init_uniform (bool, optional): initialize parameters uniformly for iterative inference, 
                default=True.
            logvar_offset (int, optional): logvar offset to init small, defaults=0.
            dropout (int, optional): rnn dropout, defaults=0.
            bidirectional (bool, optional): rnn bidirectional, defaults=False.
            seed (int, optional): random seed, defaults=0.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.z_dim = z_dim
        self.dist_type = dist_type
        self.iterative = iterative
        self.batch_size = batch_size
        self.init_uniform = init_uniform
        self.logvar_offset = logvar_offset

        output_dim = z_dim * 2
        if dist_type == "mvn":
            output_dim += z_dim **2
        
        if iterative:
            self.base = self.init_iterative(
                z_dim, batch_size, dist_type, init_uniform, logvar_offset
            )
        else:
            self.base = RNN(
                input_dim,
                output_dim, 
                hidden_dim, 
                stack_dim, 
                dropout,
                bidirectional,
            )

        if dist_type == "norm_flow":
            self.bijection = NormalizingFlow(
                z_dim, num_flows
            )
    
    def extra_repr(self):
        extra_str = ("base(dim={}, dist={}, iterative={}, batch_size={},"
        "init_uniform={}, logvar_offset={})").format(
            self.z_dim,
            self.dist_type,
            self.iterative,
            self.batch_size,
            self.init_uniform,
            self.logvar_offset
        )
        return extra_str

    def init_iterative(self, dim, batch_size, dist_type, init_uniform=True, offset=0):
        """ init iterative inference params 
        returns:
            params (torch.parameter): [batch_size, dim * 2]
        """
        if init_uniform:
            mu = 0.2 * torch.randn(1, dim).repeat(batch_size, 1)
        else:
            mu = 0.2 * torch.randn(batch_size, dim)
        
        logvar = 0.2 * torch.randn(1, dim).repeat(batch_size, 1) - offset

        if dist_type == "mvn":
            tril = torch.zeros(batch_size, dim, dim).view(batch_size, -1)
        else:
            tril = torch.empty(0)
        
        params = torch.cat([mu, logvar, tril], dim=-1)
        return nn.Parameter(params, requires_grad=True)

    def forward(self, x):
        if self.iterative:
            out = self.base
        else:
            out = self.base(x)
        return out

    def sample(self, dist_params, num_samples=1):
        P = get_mvn_dist(dist_params, self.z_dim, self.dist_type, self.logvar_offset)
        z = P.rsample((num_samples, ))
        logp = P.log_prob(z)
        if self.dist_type == "norm_flow":
            z, log_det_j = self.bijection(z)
            logp = logp - log_det_j
        return z, logp


class EBIRL(nn.Module):
    """ Empirical bayes inverse reinforcement learning """
    def __init__(
        self, 
        state_dim,
        obs_dim,
        act_dim,
        horizon,
        precision,
        rnn_hidden_dim,
        rnn_stack_dim,
        rnn_dropout,
        rnn_bidirectional=False,
        # prior settings
        prior_dist="diag",
        num_components=3,
        # post settings
        post_dist="diag",
        num_flows=5,
        iterative=False,
        batch_size=32,
        init_uniform=True,
        logvar_offset=2,
        # learning settings
        prior_lr=1e-3,
        prior_decay=0,
        prior_schedule_steps=1000,
        prior_schedule_gamma=0.5,
        vi_lr=1e-3,
        vi_decay=0,
        vi_schedule_steps=1000,
        vi_schedule_gamma=0.5,
        mc_samples=3,
        obs_penalty=0,
        kl_penalty=1,
        kl_anneal_steps=5000,
        kl_const_steps=1000,
        kl_cycles=1,
        grad_clip=20,
        seed=0
        ):
        super().__init__()
        torch.manual_seed(seed)
        self.act_dim = act_dim
        self.prior_dist = prior_dist
        self.post_dist = post_dist
        self.logvar_offset = logvar_offset
        self.num_components = num_components if prior_dist == "gmm" else 1
        self.num_flows = num_flows
        self.mc_samples = mc_samples
        self.obs_penalty = obs_penalty
        self.kl_penalty = kl_penalty 
        self.grad_clip = grad_clip
        self.seed = seed  
        
        # init agent
        self.agent = ActiveInference(
            state_dim,
            obs_dim,
            act_dim,
            horizon,
            precision, 
        )
        self.num_params = sum(self.agent.num_params)
        
        self.encoder = Encoder(
            obs_dim + act_dim + 1,
            self.num_params,
            rnn_hidden_dim,
            rnn_stack_dim,
            post_dist,
            num_flows=num_flows,
            iterative=iterative,
            batch_size=batch_size,
            init_uniform=init_uniform,
            logvar_offset=logvar_offset,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            seed=seed
        )
        
        self.prior = GaussianMixture(
            self.num_components,
            self.num_params, 
            cov="diag" if prior_dist == "gmm" else prior_dist,
            batch_norm=True,
            seed=seed
        )
        
        # init optimizers
        self.prior_optimizer = torch.optim.Adam(
            self.prior.parameters(), 
            lr=prior_lr, weight_decay=prior_decay, 
        )
        self.vi_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=vi_lr, weight_decay=vi_decay, 
        )
        
        # init scheduler 
        self.prior_scheduler = torch.optim.lr_scheduler.StepLR(
            self.prior_optimizer, 
            step_size=prior_schedule_steps, 
            gamma=prior_schedule_gamma
        )
        self.vi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.vi_optimizer, 
            step_size=vi_schedule_steps, 
            gamma=vi_schedule_gamma
        )
        self.kl_scheduler = KLScheduler(
            kl_penalty, 
            anneal_steps=kl_anneal_steps, 
            const_steps=kl_const_steps,
            num_cycles=kl_cycles
        )

    def forward(self, obs, act, mask):
        # infer latent
        obs_mask = obs * mask.unsqueeze(-1)
        obs_mask[mask.unsqueeze(-1) == 0] = -1
        act_mask = act * mask
        act_mask[mask == 0] = self.act_dim
        act_mask = F.one_hot(act_mask, self.act_dim + 1)

        z_dist = self.encoder(torch.cat([obs_mask, act_mask], dim=-1))
        z, logp_post = self.encoder.sample(z_dist, self.mc_samples)

        # decode parallel
        obs_ = obs.unsqueeze(0).repeat(self.mc_samples, 1, 1, 1).view(-1, *obs.shape[1:])
        act_ = act.unsqueeze(0).repeat(self.mc_samples, 1, 1).view(-1, *act.shape[1:])
        z_ = z.view(-1, self.num_params)

        logp_a, logp_o = self.agent(z_, obs_, act_)
        logp_a = logp_a.reshape(self.mc_samples, *act.shape).transpose(0, 1)
        logp_o = logp_o.reshape(self.mc_samples, *act.shape).transpose(0, 1)
        
        # prior for kl
        logp_prior = self.prior.log_prob(z_).view(self.mc_samples, -1) 
        return logp_a, logp_o, logp_post, logp_prior

    def evaluate(self, data_batch):
        obs, act, mask = data_batch

        # compute elbo
        logp_a, logp_o, logp_post, logp_prior = self.forward(obs, act, mask)
        kl = torch.mean(logp_post - logp_prior, dim=0)
        logp_a_mask = torch.sum(logp_a * mask.unsqueeze(1), dim=-1) 
        beta = self.kl_scheduler.compute_beta()
        elbo = torch.mean(logp_a_mask.mean(1) - beta * kl) # avg over mc samples
        
        # compute obs loss
        logp_o_mask = torch.sum(logp_o * mask.unsqueeze(1), dim=-1) 
        obs_loss = -torch.mean(logp_o_mask.mean(1))
        
        # compute total loss
        loss = -elbo + self.obs_penalty * obs_loss
        
        # collect stats 
        p_step = torch.sum(logp_a.exp() * mask.unsqueeze(1), -1) / mask.unsqueeze(1).sum(-1)

        mask_ = mask.clone()
        mask_[mask == 0] = 10000
        p_min, _ = torch.min(logp_a.exp() * mask_.unsqueeze(1), dim=-1)
        
        stats_dict = {
            "loss": loss.data.numpy(),
            "elbo": elbo.data.numpy(),  
            "logp_a": logp_a_mask.mean().data.numpy(),
            "logp_a_min": logp_a_mask.mean(1).min().data.numpy(),
            "logp_a_max": logp_a_mask.mean(1).max().data.numpy(),
            "logp_o": logp_o_mask.mean().data.numpy(),
            "logp_post": logp_post.mean().data.numpy(),
            "logp_prior": logp_prior.mean().data.numpy(),
            "kl": kl.mean().data.numpy(),
            "beta": beta.mean().numpy(),
            "p_step": p_step.mean().data.numpy(),
            "p_min": p_min.mean().data.numpy()
        }
        return loss, stats_dict
    
    def take_gradient_step(self, data_batch):
        loss, stats_dict = self.evaluate(data_batch)
        loss.backward() 
        
        nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        
        self.vi_optimizer.step()
        self.vi_optimizer.zero_grad()
        self.vi_scheduler.step()

        self.prior_optimizer.step()
        self.prior_optimizer.zero_grad()
        self.prior_scheduler.step()

        self.kl_scheduler.step()
        return stats_dict

def test_rnn_encoder():
    input_dim = 3
    z_dim = 18
    hidden_dim = 32
    stack_dim = 1
    dist_type = "mvn"
    num_flows = 3
    iterative = False
    batch_size = 10
    init_uniform = False
    logvar_offset = 0
    dropout = 0
    bidirectional = False 
    
    encoder = Encoder(
        input_dim, 
        z_dim, 
        hidden_dim, 
        stack_dim, 
        dist_type,
        num_flows,
        iterative,
        batch_size,
        init_uniform,
        logvar_offset,
        dropout,
        bidirectional
    )
    
    batch_size = 32
    T = 30
    x = torch.randn(batch_size, T, input_dim)
    dist_params = encoder(x)
    z, logp = encoder.sample(dist_params)
    
    assert list(dist_params.shape) == [batch_size, 2 * z_dim + z_dim ** 2]
    assert list(z.shape) == [1, batch_size, z_dim]
    assert list(logp.shape) == [1, batch_size]
    
    print("test rnn encoder passed")

if __name__ == "__main__":
    test_rnn_encoder()