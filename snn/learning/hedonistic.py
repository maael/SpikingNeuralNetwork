from snn.learning.method import LearningMethod

"""
Hedonistic variant of STDP Learning Method
"""
class HedonisticSTDP(LearningMethod):
  def __init__(self):
    self.adjustment = 0.1

  def update(self, layers):
    neuron.weights = neuron.weights
