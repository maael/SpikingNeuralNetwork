import numpy as np
from neurons.neuron import Neuron

"""
Leaky Integrate and Fire neuron model
"""
class LeakyIntegrateAndFireNeuron(Neuron):
  def __init__(self, weights=0):
    super(self.__class__, self).__init__(weights)
    self.degradation = 0.1
    self.refractory = False
    self.refractoryTime = 0

  def solve(self, inputs):
    self.inputs = inputs
    if self.refractoryTime > 0:
      self.refractoryTime = self.refractoryTime - 1
    else:
      self.value += np.sum(np.multiply(inputs, self.weights)) - self.degradation
    self.fire()
    if self.fired: self.refractoryTime = 4
    return self.fired