
class pretraining_param():
    img_size = 128
    lr = 3e-4
    n_epoch = 600
    batch_size = 128
class fine_tuned_param():
    img_size = 128
    batch_size = 32
    lr = 1e-4
    n_epoch = 40
    in_dim = 2048
    n_cls = 65

class hyper():
    pretraining = pretraining_param()
    fine_tuned = fine_tuned_param()