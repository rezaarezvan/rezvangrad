#include "../rezvangrad/engine.hpp"
#include <iostream>

void test_value_operations() {
  using namespace rezvangrad;

  auto a = std::make_shared<Value>(2.0);
  auto b = std::make_shared<Value>(3.0);

  std::cout << "a.value: " << a->data << ", a.grad: " << a->grad << std::endl;
  std::cout << "b.value: " << b->data << ", b.grad: " << b->grad << std::endl;
  std::cout << std::endl;

  auto c = *a + *b;
  auto d = *a + *b;
  auto e = *a * *b;
  auto f = *a / 2.0;
  auto g = a->pow(2.0);
  auto h = b->relu();

  std::cout << "c.value: " << c.data << ", c.grad: " << c.grad << std::endl;
  std::cout << "d.value: " << d.data << ", d.grad: " << d.grad << std::endl;
  std::cout << "e.value: " << e.data << ", e.grad: " << e.grad << std::endl;
  std::cout << "f.value: " << f.data << ", f.grad: " << f.grad << std::endl;
  std::cout << "g.value: " << g.data << ", g.grad: " << g.grad << std::endl;
  std::cout << "h.value: " << h.data << ", h.grad: " << h.grad << std::endl;
  std::cout << std::endl;

  c.backward();
  d.backward();
  e.backward();
  f.backward();

  std::cout << "c.grad: " << c.grad << std::endl;
  std::cout << "d.grad: " << d.grad << std::endl;
  std::cout << "e.grad: " << e.grad << std::endl;
  std::cout << "f.grad: " << f.grad << std::endl;
}

int main() {
  test_value_operations();
  return 0;
}
