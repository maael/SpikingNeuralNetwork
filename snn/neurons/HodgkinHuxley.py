import numpy as np
from snn.neurons.neuron import Neuron

"""
Hodgkin-Huxley biophysically inspired neuron model
"""
class HodgkinHuxleyNeuron(Neuron):
  def __init__(self, weights=0):
    super(self.__class__, self).__init__(weights)

  def solve(self, inputs):
    self.inputs = inputs
    self.value = np.sum(np.multiply(inputs, self.weights))
    return self.fire()