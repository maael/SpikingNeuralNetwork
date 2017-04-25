import numpy as np
from learning.method import LearningMethod

"""
Basic STDP Learning Method
"""
class STDP(LearningMethod):
  def __init__(self):
    self.adjustment = 0.1

  def update_weights(self, neuron, adjustment):
    adjustments = np.multiply(neuron.inputs, adjustment)
    signed_adjustments = np.multiply(adjustments, np.sign(neuron.weights))
    adjusted = np.add(neuron.weights, signed_adjustments)
    neuron.weights = adjusted

  def update(self, layers):
    for layer in layers:
      for neuron in layer:
        correlated_adjustment = self.adjustment if neuron.refractoryTime is 0 else -self.adjustment
        self.update_weights(neuron, correlated_adjustment)
