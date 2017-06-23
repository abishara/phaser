import math

K = 10
prob_const = {}

def set_params(_K):
  global prob_const, K
  K = _K
  prob_const.update({
    "logP(h)" : math.log(1. / K),
    "logP(err)"  : math.log(0.01),
    "logP(err')" : math.log(1 - 0.01),
  })

