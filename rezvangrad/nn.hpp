// File: rezvangrad/nn.hpp

#ifndef REZVANGRAD_NN_HPP
#define REZVANGRAD_NN_HPP

#include "engine.hpp"
#include <memory>
#include <random>
#include <vector>

namespace rezvangrad {

class Module {
public:
  virtual ~Module() = default;
  virtual void zero_grad() {
    for (auto &p : parameters()) {
      p->set_grad(0);
    }
  }
  virtual std::vector<Value *> parameters() = 0;
};

class Neuron : public Module {
public:
  Neuron(size_t nin, bool nonlin = true) : nonlin(nonlin) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    for (size_t i = 0; i < nin; ++i) {
      w.push_back(std::make_unique<Value>(dis(gen)));
    }
    b = std::make_unique<Value>(0);
  }

  Value operator()(const std::vector<Value> &x) {
    Value act(0);
    for (size_t i = 0; i < x.size(); ++i) {
      act = act + x[i] * *w[i];
    }
    act = act + *b;
    return nonlin ? act.relu() : act;
  }

  std::vector<Value *> parameters() override {
    std::vector<Value *> params;
    for (auto &param : w) {
      params.push_back(param.get());
    }
    params.push_back(b.get());
    return params;
  }

private:
  std::vector<std::unique_ptr<Value>> w;
  std::unique_ptr<Value> b;
  bool nonlin;
};

class Layer : public Module {
public:
  Layer(size_t nin, size_t nout, bool nonlin = true) {
    for (size_t i = 0; i < nout; ++i) {
      neurons.push_back(std::make_unique<Neuron>(nin, nonlin));
    }
  }

  std::vector<Value> operator()(const std::vector<Value> &x) {
    std::vector<Value> out;
    for (auto &n : neurons) {
      out.push_back((*n)(x));
    }
    return out;
  }

  std::vector<Value *> parameters() override {
    std::vector<Value *> params;
    for (auto &n : neurons) {
      auto n_params = n->parameters();
      params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
  }

private:
  std::vector<std::unique_ptr<Neuron>> neurons;
};

class MLP : public Module {
public:
  MLP(size_t nin, const std::vector<size_t> &nouts) {
    size_t sz = nin;
    for (size_t i = 0; i < nouts.size(); ++i) {
      layers.push_back(
          std::make_unique<Layer>(sz, nouts[i], i != nouts.size() - 1));
      sz = nouts[i];
    }
  }

  std::vector<Value> operator()(const std::vector<Value> &x) {
    std::vector<Value> out = x;
    for (auto &layer : layers) {
      out = (*layer)(out);
    }
    return out;
  }

  std::vector<Value *> parameters() override {
    std::vector<Value *> params;
    for (auto &layer : layers) {
      auto layer_params = layer->parameters();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
  }

private:
  std::vector<std::unique_ptr<Layer>> layers;
};

} // namespace rezvangrad

#endif // REZVANGRAD_NN_HPP
