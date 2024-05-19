#include "nn.hpp"
#include <memory>
#include <random>
#include <sstream>

namespace rezvangrad {

// Utility function to generate random values
double random_double(double min, double max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}

// Module class implementations
void Module::zero_grad() {
  for (auto &param : parameters()) {
    param->grad = 0.0;
  }
}

// Neuron class implementations
Neuron::Neuron(int nin, bool nonlin)
    : nonlin(nonlin), b(std::make_shared<Value>(0.0)) {
  for (int i = 0; i < nin; ++i) {
    w.push_back(std::make_shared<Value>(random_double(-1.0, 1.0)));
  }
}

std::shared_ptr<Value>
Neuron::operator()(const std::vector<std::shared_ptr<Value>> &x) const {
  auto act = b;
  for (size_t i = 0; i < w.size(); ++i) {
    act = std::make_shared<Value>(*act + *w[i] * *x[i]);
  }
  return nonlin ? std::make_shared<Value>(act->relu()) : act;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() const {
  std::vector<std::shared_ptr<Value>> params = w;
  params.push_back(b);
  return params;
}

std::string Neuron::repr() const {
  std::ostringstream oss;
  oss << (nonlin ? "ReLU" : "Linear") << "Neuron(" << w.size() << ")";
  return oss.str();
}

// Layer class implementations
Layer::Layer(int nin, int nout, bool nonlin) {
  for (int i = 0; i < nout; ++i) {
    neurons.emplace_back(nin, nonlin);
  }
}

std::vector<std::shared_ptr<Value>>
Layer::operator()(const std::vector<std::shared_ptr<Value>> &x) const {
  std::vector<std::shared_ptr<Value>> out;
  for (const auto &neuron : neurons) {
    out.push_back(neuron(x));
  }
  return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() const {
  std::vector<std::shared_ptr<Value>> params;
  for (const auto &neuron : neurons) {
    auto neuron_params = neuron.parameters();
    params.insert(params.end(), neuron_params.begin(), neuron_params.end());
  }
  return params;
}

std::string Layer::repr() const {
  std::ostringstream oss;
  oss << "Layer of [";
  for (const auto &neuron : neurons) {
    oss << neuron.repr() << ", ";
  }
  std::string repr = oss.str();
  repr.pop_back();
  repr.pop_back();
  repr += "]";
  return repr;
}

// MLP class implementations
MLP::MLP(int nin, const std::vector<int> &nouts) {
  size_t sz = nouts.size();
  for (size_t i = 0; i < sz; ++i) {
    layers.emplace_back(nin, nouts[i], i != sz - 1);
    nin = nouts[i];
  }
}

std::vector<std::shared_ptr<Value>>
MLP::operator()(const std::vector<std::shared_ptr<Value>> &x) const {
  std::vector<std::shared_ptr<Value>> out = x;
  for (const auto &layer : layers) {
    out = layer(out);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() const {
  std::vector<std::shared_ptr<Value>> params;
  for (const auto &layer : layers) {
    auto layer_params = layer.parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }
  return params;
}

std::string MLP::repr() const {
  std::ostringstream oss;
  oss << "MLP of [";
  for (const auto &layer : layers) {
    oss << layer.repr() << ", ";
  }
  std::string repr = oss.str();
  repr.pop_back();
  repr.pop_back();
  repr += "]";
  return repr;
}

} // namespace rezvangrad
