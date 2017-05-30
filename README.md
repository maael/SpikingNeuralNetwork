# Spiking Neural Network

This is a simple python implementation of a Spiking Neural Network (SNN) using spiking neuron models, with some Spike Timing Dependent Plasticity based learning methods.

## Requirements
- python 3

## Installation
`pip3 install -r requirements.txt`

## Neuron Models
It includes two neuron models, the simplified Leaky Integrate and Fire model and the biophysically inspired Hodgkin-Huxley model.

**Note -** Only the Integrate and Fire neuron model is implemented fully.

## Learning Methods
It includes three learning methods:
- Basic Spike Timing Dependent Plasticity (STDP)
- Homeostatic STDP variant
- Hedonistic reward-based Reinforcement Learning (RL) STDP variant

## Usage

### Include as a dependency

Add the following to your projects `requirements.txt`:
```
-e git+https://github.com/maael/SpikingNeuralNetwork.git#egg=SpikingNeuralNetwork
```

### Import components
```python
from snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from snn.learning.stdp import STDP
from snn.network.snn import SNN
```

### Create network
```python
network = SNN(total_input_neurons, [hidden_neurons_1, hidden_neurons_2], total_output_neurons, LeakyIntegrateAndFireNeuron, STDP())
```

## Tools

### Visualisation Tool
This repository also includes a simple visualisation tool that is intended to allow viewing of the network structure and firing over time.

#### usage
```
python3 snn/tools/visualise.py
```
