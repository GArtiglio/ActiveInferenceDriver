import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(
        self, 
        input_dim,
        output_dim, 
        hidden_dim, 
        stack_dim, 
        dropout=0.5,
        bidirectional=False,
        ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.stack_dim = stack_dim
        self.bidirectional = 2 if bidirectional else 1
        
        self.rnn = nn.GRU(
            input_dim, 
            hidden_dim, 
            stack_dim, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=bidirectional)
    
        self.head = make_mlp(
            self.bidirectional * self.stack_dim * hidden_dim, 
            output_dim,
            [hidden_dim, hidden_dim]
        )

    def forward(self, x):
        """
        args:
            x (torch.tensor): [batch_size, T, input_dim]

        returns:
            out (torch.tensor): [batch_size, output_dim]
        """
        batch_size = x.shape[0]

        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        hidden = hidden.transpose(0, 1).flatten(start_dim=1)
        
        hidden = torch.relu(hidden)
        out = self.head(hidden)
        return out
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state 
        GRU has no cell state
        '''
        dim = self.bidirectional * self.stack_dim
        h0 = torch.zeros(
            dim, batch_size, self.hidden_dim
        ).to(torch.float32)
        return h0


class TDNetwork(nn.Module):
    """ Fixed horizon temporal difference network """
    def __init__(
        self, 
        horizon,
        input_dim, 
        output_dim, 
        hidden_dims, 
        layer_norm=False,
        dropout=0):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.act = nn.ReLU()

        self.layers = []
        last_dim = input_dim
        for i, h in enumerate(hidden_dims):
            self.layers.append(nn.Linear(last_dim, h))
            self.layers.append(nn.LayerNorm(h)) if layer_norm else None
            self.layers.append(self.act)
            self.layers.append(nn.Dropout(dropout)) if dropout > 0 else None
            last_dim = h
        self.layers = nn.ModuleList(self.layers)
        self.head = nn.Linear(last_dim, horizon * output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # independent finite horizon predictions
        x = self.head(x)
        x = x.view(*x.shape[:-1], self.horizon, self.output_dim)
        return x


class ReplayBuffer():
    def __init__(self, num_params, max_size):
        self.params = torch.empty(0)
        self.size = 0
        self.max_size = max_size

    def push(self, params):
        self.params = torch.cat([self.params, params], dim=0)

        if len(self.params) > self.max_size:
            self.params = self.params[:self.size]
        self.size = len(self.params)

    def sample_batch(self, batch_size):
        """ if less than buffer duplicate """
        replace = True if batch_size > self.size else False
        idx = torch.multinomial(
            torch.ones(self.size), batch_size, replace
        )
        batch = self.params[idx]

        # add noise to params
        batch = batch + torch.randn_like(batch)
        return batch
    
    
class KLScheduler():
    def __init__(self, init_beta, anneal_steps=1000, const_steps=500, num_cycles=3):
        """ Scheduler for KL annealing 
        args:
            init_beta (float): initial kl constant 
            anneal_steps (int): number of kl annealing steps, 
                if None then use init_beta throughout, default=1000
            const_steps (int): number of kl=1 training steps, default=500
            num_cycles (int): number of kl anneal cycles, beta=1 after last cycle, default=3
        """
        self.init_beta = init_beta
        self.anneal_steps = anneal_steps
        self.const_steps = const_steps
        self.num_cycles = num_cycles
        self.counter = 0
        self.cycle = 0
    
    def compute_beta(self):
        if self.anneal_steps is None:
            beta = self.init_beta * torch.ones(1)
        else:
            beta = torch.ones(1)
            if self.counter < self.anneal_steps and self.cycle < self.num_cycles:
                beta = self.init_beta + (1 - self.init_beta) * self.counter / self.anneal_steps
                beta  = torch.clip(beta * torch.ones(1), 0, 1)
        return beta

    def step(self):
        self.counter += 1
        if self.anneal_steps is not None:
            if self.counter >= (self.anneal_steps + self.const_steps):
                self.counter = 0
                self.cycle += 1


def make_mlp(input_dim, output_dim, hidden_dims):
    mlp = []
    last_dim = input_dim
    for h in hidden_dims:
        mlp.append(nn.Linear(last_dim, h))
        mlp.append(nn.LeakyReLU())
        last_dim = h
    mlp.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*mlp)