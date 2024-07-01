#ifndef REZVANGRAD_ENGINE_HPP
#define REZVANGRAD_ENGINE_HPP

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace rezvangrad {

class Value : public std::enable_shared_from_this<Value> {
private:
  float _data;
  float _grad;
  std::function<void()> _backward;
  std::unordered_set<std::shared_ptr<Value>> _previous;
  std::string _operation;

public:
  Value(float data, std::unordered_set<std::shared_ptr<Value>> prev = {},
        std::string op = "")
      : _data(data), _grad(0.0), _previous(std::move(prev)),
        _operation(std::move(op)) {
    _backward = [this] {
      for (const auto &child : this->_previous) {
        child->_backward();
      }
    };
  }

  void set_grad(float grad_value) { this->_grad = grad_value; }
  float get_data() { return _data; }

  void set_data(float data) { this->_data = data; }
  float get_grad() const { return _grad; }

  std::unordered_set<std::shared_ptr<Value>> get_prev() const {
    return _previous;
  }

  std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &other) {
    auto out_prev =
        std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(_data + other->_data, out_prev, "+");
    out->_backward = [this, other, out] {
      _grad += out->_grad;
      other->_grad += out->_grad;
    };
    return out;
  }

  std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &other) {
    return (*this) + (-(*other));
  }

  std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &other) {
    auto out_prev =
        std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(_data * other->_data, out_prev, "*");
    out->_backward = [this, other, out] {
      _grad += other->_data * out->_grad;
      other->_grad += _data * out->_grad;
    };
    return out;
  }

  std::shared_ptr<Value> operator-() {
    return (*this) * std::make_shared<Value>(-1);
  }

  std::shared_ptr<Value> pow(const std::shared_ptr<Value> &other) {
    auto out_prev =
        std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out =
        std::make_shared<Value>(std::pow(_data, other->_data), out_prev, "^");
    out->_backward = [this, other, out] {
      _grad += other->_data * std::pow(_data, other->_data - 1) * out->_grad;
    };
    return out;
  }

  std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &other) {
    return (*this) * other->pow(std::make_shared<Value>(-1));
  }

  std::shared_ptr<Value> relu() {
    auto out_prev =
        std::unordered_set<std::shared_ptr<Value>>{shared_from_this()};
    auto out = std::make_shared<Value>(std::max(0.0f, _data), out_prev, "relu");
    out->_backward = [this, out] { _grad += (_data > 0) ? out->_grad : 0; };
    return out;
  }

  void backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;

    std::function<void(const std::shared_ptr<Value> &)> build_topo =
        [&](const std::shared_ptr<Value> &v) {
          if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto &child : v->_previous) {
              build_topo(child);
            }
            topo.push_back(v);
          }
        };

    build_topo(shared_from_this());
    _grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      const auto &v = *it;
      v->_backward();
    }
  }
};

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return (*lhs) + rhs;
}

inline std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return (*lhs) - rhs;
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return (*lhs) * rhs;
}

inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &lhs,
                                        const std::shared_ptr<Value> &rhs) {
  return (*lhs) / rhs;
}

inline std::shared_ptr<Value> pow(const std::shared_ptr<Value> &lhs,
                                  const std::shared_ptr<Value> &rhs) {
  return lhs->pow(rhs);
}

inline std::shared_ptr<Value> relu(const std::shared_ptr<Value> &v) {
  return v->relu();
}

} // namespace rezvangrad

#endif
