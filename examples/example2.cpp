// In your main.cpp or wherever you're testing

#include "../rezvangrad/engine.hpp"
#include "../rezvangrad/nn.hpp"
#include <iostream>

int main() {
  using namespace rezvangrad;

  // Create a simple MLP with 3 inputs, one hidden layer of 4 neurons, and 1
  // output
  MLP mlp(3, {4, 1});

  // Create some dummy input
  std::vector<Value> x = {Value(1.0), Value(2.0), Value(3.0)};

  // Perform a forward pass
  auto output = mlp(x);

  // Print the output
  std::cout << "Output: " << output[0].get_data() << std::endl;

  // Compute some dummy loss
  Value loss = output[0].pow(2);

  // Backward pass
  loss.backward();

  // Print some gradients
  auto params = mlp.parameters();
  std::cout << "First parameter gradient: " << params[0]->get_grad()
            << std::endl;

  return 0;
}
