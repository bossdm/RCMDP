



def lr_proportional(n):
    n = float(1 + n//500)
    return 1. /(n)    # stochastic approximation at two time scales (see Borkar, 2009)
def lr_sixfifths(n):
    n = float(1 + n//500)
    return 1. / ((n) ** 1.2)
def lr_dummy(_n):
    return 1.0