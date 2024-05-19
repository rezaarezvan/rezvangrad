#include "../rezvangrad/nn.hpp"
#include <iostream>

void test_neural_network() {
  using namespace rezvangrad;

  // Create a simple MLP with 2 input neurons, one hidden layer of 3 neurons,
  // and 1 output neuron
  MLP mlp(2, {3, 1});

  // Create input data
  auto x1 = std::make_shared<Value>(1.0);
  auto x2 = std::make_shared<Value>(2.0);
  std::vector<std::shared_ptr<Value>> input = {x1, x2};

  // Forward pass
  auto output = mlp(input);

  // Perform backward pass
  output[0]->backward();

  // Print the output value and gradients
  std::cout << "Output value: " << output[0]->data
            << ", Output grad: " << output[0]->grad << std::endl;

  // Print gradients of the inputs
  std::cout << "x1 grad: " << x1->grad << std::endl;
  std::cout << "x2 grad: " << x2->grad << std::endl;

  // Print parameters and their gradients
  auto params = mlp.parameters();
  for (size_t i = 0; i < params.size(); ++i) {
    std::cout << "Parameter " << i << ": " << params[i]->data
              << ", grad: " << params[i]->grad << std::endl;
  }
}

int main() {
  test_neural_network();
  return 0;
}
