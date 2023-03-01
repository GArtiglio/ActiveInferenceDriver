import torch

class Simulator:
    """ Simulator for disengagement time """
    def __init__(self, obs, act, obs_min, t_mean, t_std):
        """
        obs (torch.tensor): ground truth normalized relative 
            distance observation sequence. size=[seq_len]
        act (torch.tensor): ground truth action sequence. size=[seq_len]
        obs_min (float): minimum normalized observation
        t_mean (float): time normalization mean
        t_std (float): time normalization std
        """
        self.obs_min = obs_min # zero distance
        self.t_mean = t_mean # constant for time normalization
        self.t_std = t_std # constant for time normalization

        engage_act = torch.diff(act)
        self.t_disengage = torch.where(engage_act == -1)[0] + 1
        
        # only keep engagaed observation trajectory
        self.obs_engage = obs[:self.t_disengage]
        self.obs_grad = torch.gradient(obs)[0][:self.t_disengage] # observation gradient for interpolation

    def reset(self):
        self.t = 0
        time_since_act = (self.t - self.t_mean) / self.t_std
        return torch.tensor([self.obs_engage[self.t], time_since_act]).to(torch.float32)

    def step(self, act):
        self.t += 1
        if self.t < self.t_disengage:
            obs = self.obs_engage[self.t]
        else:
            obs = torch.clip(
                self.obs_engage[-1] + self.obs_grad[-1] * (self.t - self.t_disengage), self.obs_min, torch.inf
            )
        
        # pack 2d observation
        time_since_act = (self.t - self.t_mean) / self.t_std
        obs = torch.tensor([obs, time_since_act]).to(torch.float32)

        done = act == 0 # act 0 for disengaging

        return obs, done


def simulate(env, agent, q, max_steps):
    """ Simulate disengagement
    
    Args:
        env (Simulator): disengagement simulator
        agent (ActiveInference): active inference agent
        q (torch.tensor): parameter vector. size=[num_params]
        max_steps (int): max simulation steps

    Returns:
        obs_sim (torch.tensor): simulated observations. size=[seq_len, obs_dim]
        act_sim (torch.tensor): simulated actions. size=[seq_len]
        pi (torch.tensor): agent action probabilities. size=[seq_len, 1, act_dim]
        b (torch.tensor): agent beliefs. size=[seq_len, 1, state_dim]
    """
    obs = env.reset()

    obs_sim = [obs]
    act_sim = [torch.ones(1)] # dummy initial engage action
    for i in range(max_steps):
        with torch.no_grad():
            pi, b, _ = agent.forward(
                torch.stack(obs_sim).unsqueeze(-2),
                torch.Tensor(act_sim).long().unsqueeze(-1),
                q.unsqueeze(0)
            )
            act = torch.multinomial(pi[-1], 1)[0]
        
        obs, done = env.step(act)
        obs_sim.append(obs)
        act_sim.append(act)

        if done:
            break

    obs_sim = torch.stack(obs_sim)
    act_sim = torch.Tensor(act_sim)
    return obs_sim, act_sim, pi.squeeze(-2), b.squeeze(-2)