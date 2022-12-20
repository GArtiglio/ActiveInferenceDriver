import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from src.distributions import rectify
from src.distributions import poisson_pdf, kl_divergence

class ActiveInference(nn.Module):
    def __init__(self, state_dim, act_dim, horizon, obs_dist="norm", a_cum=False):
        super().__init__()
        assert obs_dist in ["norm", "lognorm"]
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.eps = 1e-6
        self.obs_dist = obs_dist
        self.a_cum = a_cum # cumulative mean parameterization

        # count params
        self.A_params = state_dim  * 2 # observation
        self.B_params = act_dim * state_dim ** 2 # transition
        self.C_params = state_dim # target distribution
        self.D_params = state_dim # initial belief
        self.tau_params = 1 # plan horizon
        self.beta_params = 1 # precision

        self.num_params = [
            self.A_params, self.B_params, 
            self.C_params, self.D_params, 
            self.tau_params, self.beta_params
        ]

    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, horizon={}, obs_dist={}, a_cum={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.horizon, self.obs_dist, self.a_cum
        )
        return s
    
    def transform_params(self, theta):
        """
        Returns:
            A (torch.tensor): obs distribution object.
            B (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
            C (torch.tensor): target distribution. size=[batch_size, 1, state_dim]
            D (torch.tensor): initial belief. size=[batch_size, state_dim]
            tau (torch.tensor): plan horizon poisson rate. size=[batch_size, 1]
            beta (torch.tensor): precisions. size=[horizon, batch_size]
        """
        theta = torch.split(theta, self.num_params, dim=-1)

        if self.a_cum: # cumulative parameterization to ensure mean ordering
            A_offset, A_lv = torch.chunk(theta[0], 2, dim=-1) 
            A_mean = torch.cat([A_offset[:, :1], rectify(A_offset[:, 1:]).cumsum(-1)], dim=-1)
        else:
            A_mean, A_lv = torch.chunk(theta[0], 2, dim=-1) 

        A_std = rectify(A_lv)
        A = self.get_obs_dist(A_mean, A_std)
        
        B = torch.softmax(
            theta[1].view(-1, self.act_dim, self.state_dim, self.state_dim)
        , dim=-1)
        C = torch.softmax(
            theta[2].view(-1, 1, self.state_dim), dim=-1
        )
        D = torch.softmax(
            theta[3].view(-1, self.state_dim), dim=-1
        )
        tau = self.get_poisson_rate(theta[4])
        beta = rectify(theta[5])
        beta = torch.repeat_interleave(beta, self.horizon, dim=1).T
        return A, B, C, D, tau, beta
    
    def get_obs_dist(self, A_mean, A_std):
        if self.obs_dist == "lognorm":
            dist = torch_dist.LogNormal(A_mean, A_std)
        elif self.obs_dist == "norm":
            dist = torch_dist.Normal(A_mean, A_std)
        else:
            raise ValueError
        return dist
    
    def get_poisson_rate(self, tau):
        tau_ = torch.sigmoid(tau) * self.horizon * 2 + 1
        return tau_

    def compute_efe(self, A, B, C):
        """ Compute negative expected free energy reward
        
        Args:
            A (torch.dist): obs distribution object. 
            B (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
            C (torch.tensor): target distribution. size=[batch_size, 1, state_dim]

        Returns:
            r (torch.tensor): negative efe reward. size=[batch_size, act_dim, state_dim]
        """
        ent = A.entropy()
        kl = kl_divergence(B, C.unsqueeze(1))
        eh = torch.einsum("nkij, nj -> nki", B, ent)
        r = -kl - eh
        return r
    
    def value_iteration(self, reward, transition):
        """ Finite horizon value iteration planning

        Args:
            reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
            transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
        
        Returns:
            q (torch.tensor): state action values. size=[horizon, batch_size, act_dim, state_dim]
        """
        q = [reward] + [torch.empty(0)] * (self.horizon - 1)
        for t in range(self.horizon - 1):
            v = torch.logsumexp(q[t], dim=-2)
            ev = torch.einsum("nkij, nj -> nki", transition, v)
            q[t+1] = reward + ev
        q = torch.stack(q)
        return q

    def forward(self, o, a, theta, detach=False):
        """
        Args:
            o (torch.tensor): obs sequence. size=[T, batch_size]
            a (torch.long): act sequence. size=[T, batch_size]
            theta (torch.tensor). agent params. size=[batch_size, num_params]

        Returns:
            pi (torch.tensor): policy distribution. size=[T, batch_size, act_dim]
            b (torch.tensor): belief distribution. size=[T, batch_size, state_dim]
            beta (torch.tensor): precisions. size=[T, horizon, batch_size]
            logp_o (torch.tensor): obs predictive likelihood. size=[T, batch_size]
        """
        T = o.shape[0]
        
        A, B, C, D, tau, beta0 = self.transform_params(theta)
        r = self.compute_efe(A, B, C)
        q = self.value_iteration(r, B)
        
        pi = [torch.empty(0)] * T # policy
        b = [D] + [torch.empty(0)] * T # belief
        beta = [beta0] + [torch.empty(0)] * T # precision
        logp_o = [torch.empty(0)] * T # obs predictive likelihood
        for t in range(T):
            if detach:
                pi[t] = self.compute_policy(b[t].data, q, tau, beta[t])
            else:
                pi[t] = self.compute_policy(b[t], q, tau, beta[t])
            b[t+1], logp_o[t] = self.update_belief(b[t], a[t], o[t], A, B)
            beta[t+1] = self.update_beta(beta[t], b[t+1], b[t], q)
        
        pi = torch.stack(pi)
        b = torch.stack(b[:-1])
        beta = torch.stack(beta[:-1])
        logp_o = torch.stack(logp_o)
        return pi, b, beta, logp_o

    def update_belief(self, b, a, o, A, B):
        """ Compute posterior belief and obs predictive likelihood
        
        Args:
            b (torch.tensor): current belief vector. size=[batch_size, state_dim]
            a (torch.long): current action. size=[batch_size]
            o (torch.tensor): next observation. size=[batch_size]
            A (torch.dist): obs distribution object. 
            B (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]

        Returns:
            b_next (torch.tensor): next belief vector. size=[batch_size, state_dim]
            logp_o (torch.tensor): obs predictive likelihood. size=[batch_size]
        """
        a_ = F.one_hot(a, num_classes=self.act_dim).to(torch.float32)
        s = torch.einsum("nkij, nk, ni ->nj", B, a_, b)
        
        logp_s = torch.log(s + self.eps)
        logp_o = A.log_prob(o.unsqueeze(-1))
        b_next = torch.softmax(logp_s + logp_o, dim=-1)

        logp_o = torch.logsumexp(logp_s + logp_o, dim=-1)
        return b_next, logp_o

    def update_beta(self, beta, b, b_last, q):
        """ Update precision 

        Args:
            beta (torch.tensor): current precisions. size=[horizon, batch_size]
            b (torch.tensor): current belief. size=[batch_size, state_dim]
            b_last (torch.tensor): previous belief. size=[batch_size, state_dim]
            q (torch.tensor): state action values. size=[horizon, batch_size, act_dim, state_dim]

        Returns:
            beta_next (torch.tensor): updated precisions. size=[horizon, batch_size]
        """
        v = torch.logsumexp(torch.einsum("hnki, ni -> hnk", q, b), dim=-1)
        v_last = torch.logsumexp(torch.einsum("hnki, ni -> hnk", q, b_last), dim=-1)
        beta_next = beta + (v - v_last)
        return beta_next

    def compute_policy(self, b, q, tau, beta):
        """ Compute belief-action policy

        Args:
            b (torch.tensor): current belief. size=[batch_size, state_dim]
            q (torch.tensor): state action values. size=[horizon, batch_size, act_dim, state_dim]
            tau (torch.tensor): plan horizon poisson rate. size=[batch_size, 1]
            beta (torch.tensor): precisions. size=[horizon, batch_size]

        Returns:
            pi (torch.tensor): policy distribution. size=[batch_size, act_dim]
        """
        beta_ = beta.clip(1e-6, 1e6).unsqueeze(-1)
        q_b = torch.einsum("ni, hnki -> hnk", b, q)
        pi = torch.softmax(q_b / beta_, dim=-1)
        
        h = poisson_pdf(tau, self.horizon)
        pi = torch.einsum("hnk, nh -> nk", pi, h)
        return pi

    def compute_act_loss(self, o, a, forward_out, mask):
        pi, _, _, _ = forward_out

        a_ = F.one_hot(a, num_classes=self.act_dim).to(torch.float32)
        logp_a = torch.einsum("tnk, tnk -> tn", torch.log(pi + self.eps), a_)
        
        loss = -torch.sum(logp_a * mask, dim=0) / (mask.sum(0) + self.eps)
        stats = {"act_loss": loss.mean().data.item()}
        return loss, stats

    def compute_obs_loss(self, o, a, forward_out, mask):
        _, _, _, logp_o = forward_out

        loss = -torch.sum(logp_o * mask, dim=0) / (mask.sum(0) + self.eps)
        stats = {"obs_loss": loss.mean().data.item()}
        return loss, stats

if __name__ == "__main__":
    torch.manual_seed(0)
    state_dim = 10
    act_dim = 3
    horizon = 30
    agent = ActiveInference(
        state_dim, act_dim, horizon
    )
    
    # synthetic data
    T = 24
    batch_size = 12
    o = torch.randn(T, batch_size).abs()
    a = torch.randint(0, 2, (T, batch_size))
    mask = torch.randint(0, 2, (T, batch_size))
    theta = nn.Parameter(torch.randn(batch_size, sum(agent.num_params)))

    out = agent(o, a, theta)
    act_loss, act_stats = agent.compute_act_loss(o, a, out, mask)
    obs_loss, obs_stats = agent.compute_obs_loss(o, a, out, mask)
    
    # grad check
    act_loss.mean().backward()
    print("num zero grad", torch.sum(theta.grad.data == 0))
    print("num nan grad", torch.sum(theta.grad.data == torch.nan))