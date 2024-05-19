#include "engine.hpp"
#include <cmath>

namespace rezvangrad {

Value::Value(double data, std::set<std::shared_ptr<Value>> children,
             std::string op)
    : data(data), grad(1.0), _backward([]() {}), _prev(children), _op(op) {}

Value Value::operator+(const Value &other) const {
  auto out = std::make_shared<Value>(
      this->data + other.data,
      std::set<std::shared_ptr<Value>>{std::make_shared<Value>(*this),
                                       std::make_shared<Value>(other)},
      "+");

  out->_backward = [out, this_shared = std::make_shared<Value>(*this),
                    other_shared = std::make_shared<Value>(other)]() {
    this_shared->grad += out->grad;
    other_shared->grad += out->grad;
  };

  return *out;
}

Value Value::operator*(const Value &other) const {
  auto out = std::make_shared<Value>(
      this->data * other.data,
      std::set<std::shared_ptr<Value>>{std::make_shared<Value>(*this),
                                       std::make_shared<Value>(other)},
      "*");

  out->_backward = [out, this_shared = std::make_shared<Value>(*this),
                    other_shared = std::make_shared<Value>(other)]() {
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
  auto out = std::make_shared<Value>(
      std::pow(this->data, exponent),
      std::set<std::shared_ptr<Value>>{std::make_shared<Value>(*this)},
      "**" + std::to_string(exponent));

  out->_backward = [out, this_shared = std::make_shared<Value>(*this),
                    exponent]() {
    this_shared->grad +=
        (exponent * std::pow(this_shared->data, exponent - 1)) * out->grad;
  };

  return *out;
}

Value Value::relu() const {
  auto out = std::make_shared<Value>(
      std::max(0.0, this->data),
      std::set<std::shared_ptr<Value>>{std::make_shared<Value>(*this)}, "ReLU");

  out->_backward = [out, this_shared = std::make_shared<Value>(*this)]() {
    this_shared->grad += (out->data > 0) * out->grad;
  };

  return *out;
}

void Value::backward() {
  std::vector<std::shared_ptr<Value>> topo;
  std::set<std::shared_ptr<Value>> visited;
  build_topo(topo, visited);

  this->grad = 1.0;
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->_backward();
  }
}

void Value::build_topo(std::vector<std::shared_ptr<Value>> &topo,
                       std::set<std::shared_ptr<Value>> &visited) {
  if (visited.find(std::make_shared<Value>(*this)) == visited.end()) {
    visited.insert(std::make_shared<Value>(*this));
    for (auto child : this->_prev) {
      child->build_topo(topo, visited);
    }
    topo.push_back(std::make_shared<Value>(*this));
  }
}

} // namespace rezvangrad
