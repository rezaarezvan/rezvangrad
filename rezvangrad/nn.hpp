#ifndef REZVANGRAD_NN_HPP
#define REZVANGRAD_NN_HPP

#include "engine.hpp"
#include <memory>
#include <string>
#include <vector>

namespace rezvangrad {

class Module {
public:
  virtual std::vector<std::shared_ptr<Value>> parameters() const = 0;
  void zero_grad();
};

class Neuron : public Module {
public:
  Neuron(int nin, bool nonlin = true);

  std::shared_ptr<Value>
  operator()(const std::vector<std::shared_ptr<Value>> &x) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;

  std::string repr() const;

private:
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> b;
  bool nonlin;
};

class Layer : public Module {
public:
  Layer(int nin, int nout, bool nonlin = true);

  std::vector<std::shared_ptr<Value>>
  operator()(const std::vector<std::shared_ptr<Value>> &x) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;

  std::string repr() const;

private:
  std::vector<Neuron> neurons;
};

class MLP : public Module {
public:
  MLP(int nin, const std::vector<int> &nouts);

  std::vector<std::shared_ptr<Value>>
  operator()(const std::vector<std::shared_ptr<Value>> &x) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;

  std::string repr() const;

private:
  std::vector<Layer> layers;
};

} // namespace rezvangrad

#endif // REZVANGRAD_NN_HPP
