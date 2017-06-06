import numpy as np
from snn.learning.stdp import STDP

"""
Hedonistic variant of STDP Learning Method
"""
class HedonisticSTDP(STDP):
  def __init__(self):
    super(self.__class__, self).__init__()
    self.tau_c = 1
    self.tau_d = 0.2
    self.d = 0
    self.reward = 0.1

  def setup(self, neuron):
    super(self.__class__, self).setup(neuron)
    neuron.c = np.zeros(len(neuron.inputs))

  def calculate_c(self, neuron, stdp_weights):
    delta = 1
    neuron.c = np.add(np.divide(neuron.c, self.tau_c), np.multiply(stdp_weights, delta))

  def calculate_s(self, neuron):
    return np.add(np.multiply(neuron.c, self.d), neuron.weights)

  def calculate_d(self):
    self.d = (self.d / self.tau_d) + self.reward

  def update(self, layers):
    for layer in layers:
      for neuron in layer:
        if (not self.is_setup):
          self.setup(neuron)
        correlated_adjustment = self.adjustment if neuron.refractoryTime is 0 else -self.adjustment
        stdp_weight_updates = self.update_weights(neuron, correlated_adjustment)
        self.calculate_c(neuron, stdp_weight_updates)
        self.calculate_d()
        neuron.weights = self.calculate_s(neuron)
    if (not self.is_setup): self.is_setup = True
    self.time = self.time + 1
