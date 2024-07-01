// File: examples/example.cpp

#include "../rezvangrad/engine.hpp"
#include <iostream>

int main() {
  using namespace rezvangrad;

  // Create some values
  Value a(2.0);
  Value b(3.0);

  // Perform operations
  Value c = a + b;
  Value d = a * b;
  Value e = c.relu();
  Value f = d.pow(2);

  // Compute gradients
  f.backward();

  // Print results
  std::cout << "a: data = " << a.get_data() << ", grad = " << a.get_grad()
            << std::endl;
  std::cout << "b: data = " << b.get_data() << ", grad = " << b.get_grad()
            << std::endl;
  std::cout << "c: data = " << c.get_data() << ", grad = " << c.get_grad()
            << std::endl;
  std::cout << "d: data = " << d.get_data() << ", grad = " << d.get_grad()
            << std::endl;
  std::cout << "e: data = " << e.get_data() << ", grad = " << e.get_grad()
            << std::endl;
  std::cout << "f: data = " << f.get_data() << ", grad = " << f.get_grad()
            << std::endl;

  return 0;
}
