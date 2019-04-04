from keras.optimizers import Adam, RMSprop, SGD, Adagrad

def optimizer(hps):
    """Set up a keras optimizer"""
    if hps['optimizer']=='Adam':
        opt = Adam(lr=hps['learning_rate'],
                   beta_1 = hps['opt_beta_1'] if 'opt_beta_1' in hps else 0.9,
                   beta_2 = hps['opt_beta_2'] if 'opt_beta_2' in hps else 0.999,
                   decay=hps['opt_decay'] if 'opt_decay' in hps else 0.0,
                   amsgrad=hps['opt_amsgrad'] if 'opt_amsgrad' in hps else False)
    elif hps['optimizer']  == 'SGD':
        opt = SGD(lr=hps['learning_rate'],
                      momentum=hps['opt_momentum'] if 'opt_momentum' in hps else 0.0,
                      decay=hps['opt_decay'] if 'opt_decay' in hps else 0.0)
    elif hps['optimizer'] == 'RMSprop':
        opt = RMSprop(lr=hps['learning_rate'],
                      rho=hps['opt_rho'] if 'opt_rho' in hps else 0.9,
                      decay=hps['opt_decay'] if 'opt_decay' in hps else 0.0)
    elif hps['optimizer'] == 'Adagrad':
        opt = Adagrad(lr=hps['learning_rate'],
                      decay=hps['opt_decay'] if 'opt_decay' in hps else 0.0)
    else:
        raise Exception('optimizer: invalid optimizer option')

    return opt
