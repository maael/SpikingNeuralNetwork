from learning.method import LearningMethod

"""
Basic Reinforcement Learning Method
"""
class Reinforce(LearningMethod):
  def __init__(self):
    self.adjustment = 0.1

  def update(self, layers):
    neuron.weights = neuron.weights
