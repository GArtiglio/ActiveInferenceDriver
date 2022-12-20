import math
import torch
import torch.nn as nn
import torch.distributions as dist

class ActiveInference(nn.Module):
    """ Active inference agent
    Discrete states and actions
    Diagonal lognormal observations

    The agent takes in a list of parameters and a sequence 
    of observations and applys the following steps:
        1. Apply corresponding transformations to the parameters
        2. Update state belief based on observation
        3. Calculate action probability 
        4. Repeat steps 2 and 3 until the last observation
    
    Agent parameters are described in transform_params()
    """
    def __init__(
        self, 
        state_dim,
        obs_dim,
        act_dim,
        horizon,
        precision, 
        seed=0
        ):
        """
        args:
            state_dim (int): state dimension
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            horizon (int): max planning horizon
            precision (int): fit precision
            seed (int, optional): random seed. Defaults to 0.
        """
        super().__init__()
        assert precision == 0 or precision == 1, "Invalid precision (0 or 1)" 
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.precision = precision # fit precision
        self.tau = 1 # fit horizon
        self.eps = 1e-6
        self.seed = seed

        # count params
        self.A_params = state_dim * obs_dim * 2 # observation
        self.B_params = act_dim * state_dim ** 2 # transition
        self.C_params = state_dim # reward
        self.D_params = state_dim # initial belief

        self.num_params = [
            self.A_params, self.B_params, 
            self.C_params, self.D_params, self.tau, self.precision
        ]
    
    def extra_repr(self):
        """ print function """
        extra_str = "agent(state_dim={}, obs_dim={}, act_dim={}, horizon={}, precision={})".format(
            self.state_dim,
            self.obs_dim,
            self.act_dim,
            self.horizon,
            True if self.precision == 1 else False,
        )
        return extra_str

    def forward(self, params, obs, act):
        """ 
        args:
            params: (torch.float32) [batch_size, num_params]
            obs: (torch.float32) [batch_size, T, obs_dim]
            act: (torch.long) [batch_size, T]
        """
        batch_size = len(obs)
        T = obs.shape[1]
        (A_mean, A_std, B, C, D, inv_tau, inv_beta) = self.transform_params(params)
        tau = 1 / torch.clip(inv_tau, 1e-6, 1e6) 
        beta = 1 / inv_beta.unsqueeze(-2).repeat(1, self.horizon, 1)
        gamma = 1 / torch.clip(beta, 1e-6, 1e6)
        h_dist = poisson_pdf(tau, self.horizon)
        Q = self.backward_recursion([A_mean, A_std, B, C]) 
        
        # rollout
        beliefs = [torch.empty(0)] * (T + 1)
        beliefs[0] = D 
        act_probs = [torch.empty(0)] * T
        obs_probs = [torch.empty(0)] * T
        for t in range(T):
            # compute finite horizion action probs using soft Q
            act_probs[t] = self.compute_action_dist(beliefs[t], Q, h_dist, gamma) 

            # update belief
            beliefs[t+1] = self.update_belief(
                beliefs[t], 
                obs[:, t], 
                act[:, t], 
                [A_mean, A_std, B]
            )
            
            # update precision
            if t > 0:
                G = self.compute_EFE(beliefs[t-1], Q)
                diff_pi = act_probs[t] - act_probs[t-1]
                beta = beta + torch.sum(diff_pi.unsqueeze(-2) * G, dim=-1, keepdim=True)
                gamma = 1 / torch.clip(beta, 1e-6, 1e6)

            # compute observation probs
            obs_probs[t] = self.compute_observation_probs(beliefs[t+1], obs[:, t], [A_mean, A_std])
        
        # [batch_size, T, dim]
        beliefs = torch.stack(beliefs)[:-1].permute(1, 0, 2)
        act_probs = torch.stack(act_probs).permute(1, 0, 2)
        obs_probs = torch.stack(obs_probs).transpose(0, 1)

        logp_a = torch.log(
            torch.gather(act_probs, 2, act.view(batch_size, T, 1)).squeeze(-1) + 
            self.eps
        )
        logp_o = torch.log(obs_probs + self.eps)
        return logp_a, logp_o

    def transform_params(self, params):
        """ unpack and transform params

        args:
            params (torch.tensor): agent param vector [batch_size, num_params]

        returns:
            A_mean (torch.tensor): observation mean, no transform [batch_size, obs_dim, state_dim]
            A_std (torch.tensor): observation std, exp transform [batch_size, obs_dim, state_dim]
            B (torch.tensor): transition parameters, softmax transform [batch_size, act_dim, state_dim, state_dim]
            C (torch.tensor): reward parameters, softmax transform [batch_size, state_dim]
            D (torch.tensor): initial belief over state, softmax transform [batch_size, state_dim]
            inv_tau (torch.tensor): inverse planning horizon (bias long horizon), exp transform [batch_size, 1]
            beta (torch.tensor): precision rate, exp transform [batch_size, 1]
        """
        batch_size = len(params)
        params = [
            params[:, sum(self.num_params[:i]):sum(self.num_params[:i+1])] 
            for i in range(len(self.num_params))
        ] 
        
        A_mean, A_logvar = torch.chunk(
            params[0].view(batch_size, -1, self.state_dim), 2, dim=1
        )
        A_std = torch.exp(0.5 * A_logvar.clip(math.log(self.eps), math.log(1e5)))

        B = torch.softmax(
            params[1].view(batch_size, self.act_dim, self.state_dim, self.state_dim)
        , dim=-1)
    
        C = torch.softmax(params[2], dim=-1)

        D = torch.softmax(params[3], dim=-1)
        
        # poisson rate
        inv_tau = torch.exp(params[4].clip(math.log(1e-4), math.log(1e4)))
        
        # precision rate
        beta = torch.exp(params[5].clip(math.log(1e-4), math.log(1e4)))
        if self.precision != 1:
            beta = torch.ones(batch_size, 1)
    
        return (A_mean, A_std, B, C, D, inv_tau, beta)
    
    def update_belief(self, b, obs, act, params):
        batch_size = len(b)
        A_mean, A_std, B = params
        B_a = B[torch.arange(batch_size), act.long()]
        
        # obs likelihood
        log_A_o = dist.LogNormal(A_mean, A_std).log_prob(obs.unsqueeze(-1)).sum(1)
        
        # update belief
        s_next = torch.sum(b.unsqueeze(-1) * B_a, 1)
        b_next = torch.softmax(
            log_A_o + torch.log(s_next + self.eps), dim=-1
        )
        return b_next
    
    def compute_observation_probs(self, belief, obs, params):
        A_mean, A_std = params
        A_o = dist.LogNormal(A_mean, A_std).log_prob(obs.unsqueeze(-1)).sum(1).exp()
        obs_probs = torch.sum(belief * A_o, dim=-1)
        return obs_probs

    def backward_recursion(self, params):
        batch_size = len(params[0])
        [A_mean, A_std, B, C] = params
        C = C.view(batch_size, 1, 1, self.state_dim)
        entropy_A = torch.sum(
            dist.Normal(A_mean, A_std).entropy(), dim=1
        ).view(batch_size, 1, 1, self.state_dim)
        
        kl = torch.sum(B * (torch.log(B + self.eps) - torch.log(C + self.eps)), dim=-1)
        entropy = torch.sum(B * entropy_A, dim=-1)
        r = (kl + entropy).transpose(1, 2)        
        
        # inplace operation
        Q = [torch.empty(0)] * (self.horizon)
        Q[0] = r
        for h in range(self.horizon - 1):
            V_next = torch.logsumexp(Q[h], dim=-1)
            Q_next = torch.sum(
                B * V_next.view(batch_size, 1, self.state_dim, 1).repeat(1, self.act_dim, 1, 1)
            , dim=-2)
            Q[h + 1] = r + Q_next.transpose(1, 2)
        
        Q = torch.stack(Q).transpose(0, 1)
        return Q

    def compute_EFE(self, belief, Q):
        """ compute expected free energy using qmdp
        args: 
            belief: state belief [batch_size, state_dim]
            Q: Q value [batch_size, horizon, state_dim, act_dim]

        returns:
            G: belief action value [batch_size, horizon, act_dim]
        """
        batch_size = len(belief)
        belief = belief.view(batch_size, 1, self.state_dim, 1)
        G = torch.sum(belief * Q, dim=-2)
        return G

    def compute_action_dist(self, belief, Q, tau, gamma):
        """ compute action dist averaged over horizon dist
        
        args:
            belief (torch.tensor): state belief [batch_size, state_dim]
            Q (torch.tensor): Q value [batch_size, horizon, state_dim, act_dim]
            tau (torch.tensor): plan horizon dist [batch_size, horizon]
            gamma (torch.tensor): precision [batch_size, horizon, 1]

        returns:
            act_probs (torch.tensor): action probs [batch_size, act_dim]
        """
        G = self.compute_EFE(belief, Q)
        act_probs = torch.softmax(-gamma * G, dim=-1)
        act_probs = torch.sum(tau.unsqueeze(-1) * act_probs, dim=1)
        return act_probs


def poisson_pdf(gamma, K):
    """
    args:
        gamma (torch.float): arrival rate [batch_size, 1]
        K (int): number of classes 
    
    returns:
        pdf (torch.tensor): truncated poisson pdf [batch_size, num_classes]
    """    
    logit = torch.cat(
        [dist.Poisson(gamma).log_prob(torch.Tensor([k])).exp() for k in range(K)]
    , dim=1) + 1e-6
    
    pdf = logit / logit.sum(dim=-1, keepdim=True)
    return pdf