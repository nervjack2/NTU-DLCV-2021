# Hyperparameters

class hyper():
    # Model hyperparameters
    n_way = 5
    k_shot = 1 
    n_query = 15 
    # Training hyperparameters
    batch_size = n_way * (n_query + k_shot)
    lr = 1e-3
    n_epoch = 100
    n_eps_per_epoch = 100
    lr_scheduler_step = 20
    lr_scheduler_gamma = 0.5
    # Model architecture
    emb_dim = 1600
    
    
