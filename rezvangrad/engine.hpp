#ifndef REZVANGRAD_ENGINE_HPP
#define REZVANGRAD_ENGINE_HPP

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace rezvangrad {

class Value {
public:
  Value(double data, const std::vector<Value *> &children = {},
        const std::string &op = "");
  Value(const Value &other) = default;
  Value(Value &&other) noexcept = default;
  Value &operator=(const Value &other) = default;
  Value &operator=(Value &&other) noexcept = default;

  Value operator+(const Value &other) const;
  Value operator*(const Value &other) const;
  Value pow(double exponent) const;
  Value relu() const;
  void backward();

  double get_data() const { return _data; }
  double get_grad() const { return _grad; }
  void set_grad(double grad) { _grad = grad; }

private:
  double _data;
  mutable double _grad;
  std::function<void()> _backward;
  std::vector<Value *> _prev;
  std::string _op;

  void build_topo(std::vector<Value *> &topo,
                  std::unordered_set<Value *> &visited) const;
};

Value::Value(double data, const std::vector<Value *> &children,
             const std::string &op)
    : _data(data), _grad(0), _prev(children), _op(op) {
  _backward = []() {};
}

Value Value::operator+(const Value &other) const {
  Value out(_data + other._data,
            {const_cast<Value *>(this), const_cast<Value *>(&other)}, "+");
  out._backward = [&out, this, &other]() {
    this->_grad += out._grad;
    other._grad += out._grad;
  };
  return out;
}

Value Value::operator*(const Value &other) const {
  Value out(_data * other._data,
            {const_cast<Value *>(this), const_cast<Value *>(&other)}, "*");
  out._backward = [&out, this, &other]() {
    this->_grad += other._data * out._grad;
    other._grad += this->_data * out._grad;
  };
  return out;
}

Value Value::pow(double exponent) const {
  Value out(std::pow(_data, exponent), {const_cast<Value *>(this)},
            "**" + std::to_string(exponent));
  out._backward = [&out, this, exponent]() {
    this->_grad += exponent * std::pow(this->_data, exponent - 1) * out._grad;
  };
  return out;
}

Value Value::relu() const {
  Value out(_data < 0 ? 0 : _data, {const_cast<Value *>(this)}, "ReLU");
  out._backward = [&out, this]() {
    this->_grad += (out._data > 0) * out._grad;
  };
  return out;
}

void Value::backward() {
  std::vector<Value *> topo;
  std::unordered_set<Value *> visited;
  build_topo(topo, visited);

  _grad = 1.0;
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->_backward();
  }
}

void Value::build_topo(std::vector<Value *> &topo,
                       std::unordered_set<Value *> &visited) const {
  if (visited.find(const_cast<Value *>(this)) == visited.end()) {
    visited.insert(const_cast<Value *>(this));
    for (const auto &child : _prev) {
      child->build_topo(topo, visited);
    }
    topo.push_back(const_cast<Value *>(this));
  }
}

} // namespace rezvangrad

#endif // REZVANGRAD_ENGINE_HPP
