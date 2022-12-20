import torch
from ebirl.models import RNN, TDNetwork, ReplayBuffer, KLScheduler
from torch.nn.functional import dropout

torch.manual_seed(123)

def test_rnn():
    input_dim = 3
    output_dim = 2
    hidden_dim = 32
    stack_dim = 1
    dropout = 0.5
    bidirectional = False
    
    batch_size = 32
    T = 12
    x = torch.randn(batch_size, T, input_dim)
    
    # test simple
    rnn = RNN(
        input_dim,
        output_dim, 
        hidden_dim, 
        stack_dim, 
        dropout,
        bidirectional,
    )
    y = rnn(x)
    assert list(y.shape) == [batch_size, output_dim]
    
    # test bidirectional
    bidirectional_ = True
    rnn = RNN(
        input_dim,
        output_dim, 
        hidden_dim, 
        stack_dim, 
        dropout,
        bidirectional_,
    )
    y = rnn(x)
    assert list(y.shape) == [batch_size, output_dim]
    
    # test stacked layers
    stack_dim_ = 3
    rnn = RNN(
        input_dim,
        output_dim, 
        hidden_dim, 
        stack_dim_, 
        dropout,
        bidirectional_,
    )
    y = rnn(x)
    assert list(y.shape) == [batch_size, output_dim]
    
    print("test rnn passed")

def test_kl_scheduler():
    init_beta = 0
    anneal_steps=10
    const_steps=5
    num_cycles=3
    
    # test init beta
    kl_scheduler = KLScheduler(
        init_beta, anneal_steps, const_steps, num_cycles
    )
    beta = kl_scheduler.compute_beta()
    assert beta == init_beta
    
    init_beta_ = 0.5
    kl_scheduler = KLScheduler(
        init_beta_, anneal_steps, const_steps, num_cycles
    )
    beta = kl_scheduler.compute_beta()
    assert beta == init_beta_
    
    # test anneal steps
    total_steps = num_cycles * (anneal_steps + const_steps)
    extra_steps = 7
    for i in range(total_steps + extra_steps):
        beta = kl_scheduler.compute_beta()
        kl_scheduler.step()
        
        if (i + 1) < anneal_steps:
            assert beta < 1
        if i == anneal_steps:
            assert beta == 1
        if (i + 1) % (anneal_steps + const_steps) == 0:
            assert beta == 1
        
    print("test kl scheduler passed")

if __name__ == "__main__":
    test_rnn()
    test_kl_scheduler()