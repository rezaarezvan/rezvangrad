#ifndef REZVANGRAD_ENGINE_HPP
#define REZVANGRAD_ENGINE_HPP

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace rezvangrad {

class Value {
public:
  double data;
  double grad;
  std::function<void()> _backward;
  std::set<std::shared_ptr<Value>> _prev;
  std::string _op;

  Value(double data, std::set<std::shared_ptr<Value>> children = {},
        std::string op = "");

  Value operator+(const Value &other) const;
  Value operator*(const Value &other) const;
  Value operator-(const Value &other) const;
  Value operator/(const Value &other) const;
  Value pow(double exponent) const;
  Value relu() const;

  void backward();
  void build_topo(std::vector<std::shared_ptr<Value>> &topo,
                  std::set<std::shared_ptr<Value>> &visited);
};

} // namespace rezvangrad

#endif // REZVANGRAD_ENGINE_HPP
