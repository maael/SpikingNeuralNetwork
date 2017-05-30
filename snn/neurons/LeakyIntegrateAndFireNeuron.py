import numpy as np
from snn.neurons.neuron import Neuron

"""
Leaky Integrate and Fire neuron model
"""
class LeakyIntegrateAndFireNeuron(Neuron):
  def __init__(self, weights=0):
    super(self.__class__, self).__init__(weights)
    self.degradation = 0.9
    self.refractoryTime = 0

  def calculate_potential (self, inputs):
    self.potential = (self.potential + np.sum(np.multiply(inputs, self.weights))) * self.degradation
    return self.potential

  def solve(self, inputs):
    self.inputs = inputs
    if self.refractoryTime > 0:
      self.refractoryTime = self.refractoryTime - 1
    else:
      self.value = self.calculate_potential(inputs)
    self.fire()
    if self.fired: self.refractoryTime = 2
    return self.fired