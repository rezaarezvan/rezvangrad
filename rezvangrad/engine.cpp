#include "engine.hpp"
#include <cmath>

namespace rezvangrad {

Value::Value(double data, std::set<std::shared_ptr<Value>> children,
             std::string op)
    : data(data), grad(0.0), _backward([]() {}), _prev(children), _op(op) {}

Value Value::operator+(const Value &other) const {
  auto this_shared = const_cast<Value *>(this)->shared_from_this();
  auto other_shared = std::make_shared<Value>(other);
  auto out = std::make_shared<Value>(
      this->data + other.data,
      std::set<std::shared_ptr<Value>>{this_shared, other_shared}, "+");

  out->_backward = [this_shared, other_shared, out]() {
    this_shared->grad += out->grad;
    other_shared->grad += out->grad;
  };

  return *out;
}

Value Value::operator*(const Value &other) const {
  auto this_shared = const_cast<Value *>(this)->shared_from_this();
  auto other_shared = std::make_shared<Value>(other);
  auto out = std::make_shared<Value>(
      this->data * other.data,
      std::set<std::shared_ptr<Value>>{this_shared, other_shared}, "*");

  out->_backward = [this_shared, other_shared, out]() {
    this_shared->grad += other_shared->data * out->grad;
    other_shared->grad += this_shared->data * out->grad;
  };

  return *out;
}

Value Value::operator-(const Value &other) const {
  return *this + (other * -1);
}

Value Value::operator/(const Value &other) const {
  return *this * other.pow(-1);
}

Value Value::pow(double exponent) const {
  auto this_shared = const_cast<Value *>(this)->shared_from_this();
  auto out =
      std::make_shared<Value>(std::pow(this->data, exponent),
                              std::set<std::shared_ptr<Value>>{this_shared},
                              "**" + std::to_string(exponent));

  out->_backward = [this_shared, out, exponent]() {
    this_shared->grad +=
        (exponent * std::pow(this_shared->data, exponent - 1)) * out->grad;
  };

  return *out;
}

Value Value::relu() const {
  auto this_shared = const_cast<Value *>(this)->shared_from_this();
  auto out = std::make_shared<Value>(
      std::max(0.0, this->data), std::set<std::shared_ptr<Value>>{this_shared},
      "ReLU");

  out->_backward = [this_shared, out]() {
    this_shared->grad += (out->data > 0) * out->grad;
  };

  return *out;
}

void Value::backward() {
  // Build topological order of the graph
  std::vector<std::shared_ptr<Value>> topo;
  std::set<std::shared_ptr<Value>> visited;
  build_topo(topo, visited);

  // Initialize the gradient of the root variable
  this->grad = 1.0;

  // Backward pass
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->_backward();
  }
}

void Value::build_topo(std::vector<std::shared_ptr<Value>> &topo,
                       std::set<std::shared_ptr<Value>> &visited) {
  auto self = shared_from_this();
  if (visited.find(self) == visited.end()) {
    visited.insert(self);
    for (auto &child : this->_prev) {
      child->build_topo(topo, visited);
    }
    topo.push_back(self);
  }
}

} // namespace rezvangrad
