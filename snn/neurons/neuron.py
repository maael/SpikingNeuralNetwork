import numpy as np

"""
Basic neuron model with shared instantiation
"""
class Neuron:
  def __init__(self, weights=0):
    np.seterr(all='ignore')
    self.input = 0
    self.value = 0
    self.output = 0
    self.threshold = 0
    self.fired = False
    self.potential = 0
    self.weights = np.array([self.init_weight(weights) for x in range(weights)], dtype='float64')

  def fire(self):
    self.fired = True if (self.value > self.threshold) else False
    if self.fired:
      self.value = 0
    return 1 if self.fired else 0

  def init_weight(self, num_weights):
    return np.random.uniform(-(2 / num_weights), (2 / num_weights))

  def solve(self):
    raise NotImplementedError("A neuron model needs a solve method")
