#ifndef REZVANGRAD_NN_HPP
#define REZVANGRAD_NN_HPP

#include "engine.hpp"

#include <iostream>
#include <random>
#include <vector>

namespace rezvangrad {

class Module {
public:
  void zero_grad() {
    for (auto &weight : parameters()) {
      weight->set_grad(0.0);
    }
  }
  virtual std::vector<std::shared_ptr<Value>> parameters() = 0;
};

class Neuron : public Module {
private:
  std::vector<std::shared_ptr<Value>> _weights;
  std::shared_ptr<Value> _biase = std::make_shared<Value>(0);
  bool _nonlin;

public:
  Neuron(int nin, bool nonlin = true) : _nonlin(nonlin) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    _weights.reserve(nin);
    for (int i = 0; i < nin; ++i) {
      auto weight = std::make_shared<Value>(dis(gen));
      _weights.emplace_back(weight);
    }
  }

  std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> &x) {
    std::shared_ptr<Value> act = std::make_shared<Value>(0.0);
    for (int i = 0; i < x.size(); ++i) {
      act = act + (x[i] * _weights[i]);
    }
    act = act + _biase;
    return _nonlin ? act->relu() : act;
  }

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> parameters;
    parameters.reserve(_weights.size() + 1);
    for (auto &weight : _weights) {
      parameters.emplace_back(weight);
    }
    parameters.emplace_back(_biase);
    return parameters;
  }

  void show_parameters() {
    std::cout << "weights: ";
    for (auto &weight : _weights) {
      std::cout << weight->get_data() << ", ";
    }
    std::cout << "bias: " << _biase->get_data() << std::endl;
  }
};

class Layer : public Module {
private:
  std::vector<Neuron> _neurons;
  int _total_parameters;

public:
  Layer(int nin, int nout) : _total_parameters((nin + 1) * nout) {
    _neurons.reserve(nout);
    for (int i = 0; i < nout; ++i) {
      _neurons.emplace_back(Neuron(nin, true));
    }
  }

  std::vector<std::shared_ptr<Value>>
  operator()(std::vector<std::shared_ptr<Value>> x) {
    std::vector<std::shared_ptr<Value>> out;
    out.reserve(_neurons.size());
    for (auto &neuron : _neurons) {
      out.emplace_back(neuron(x));
    }
    return out;
  }

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> parameters;
    parameters.reserve(_total_parameters);
    for (auto &neuron : _neurons) {
      for (auto &weight : neuron.parameters()) {
        parameters.emplace_back(weight);
      }
    }
    return parameters;
  }

  void show_parameters() {
    std::cout << "Layer Weights: " << _total_parameters << std::endl;
    for (auto &neuron : _neurons) {
      neuron.show_parameters();
    }
  }
};

class MLP : public Module {
private:
  std::vector<Layer> _layers;
  int _total_parameters;

public:
  MLP(int nin, std::vector<int> nout) : _total_parameters(0) {
    _layers.reserve(nout.size());
    for (int i = 0; i < nout.size(); ++i) {
      if (i == 0) {
        _layers.emplace_back(Layer(nin, nout[i]));
        _total_parameters += nin * nout[i];
      } else {
        _layers.emplace_back(Layer(nout[i - 1], nout[i]));
        _total_parameters += nout[i - 1] * nout[i];
      }
    }
  }

  std::vector<std::shared_ptr<Value>>
  operator()(std::vector<std::shared_ptr<Value>> x) {
    for (auto &layer : _layers) {
      x = layer(x);
    }
    return x;
  }

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> parameters;
    parameters.reserve(_total_parameters);
    for (auto &layer : _layers) {
      for (auto &param : layer.parameters()) {
        parameters.emplace_back(param);
      }
    }
    return parameters;
  }

  void show_parameters() {
    int i = 0;
    for (auto &layer : _layers) {
      std::cout << "\nLayer" << i << ": " << std::endl;
      layer.show_parameters();
      i++;
    }
  }
};

} // namespace rezvangrad

#endif
