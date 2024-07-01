#include "../rezvangrad/engine.hpp"
#include "../rezvangrad/nn.hpp"
#include <iostream>
#include <vector>

using namespace rezvangrad;

int main() {
  MLP mlp(3, {4, 1});

  auto params = mlp.parameters();
  std::cout << "Initial parameters:" << std::endl;
  for (size_t i = 0; i < params.size(); ++i) {
    std::cout << "Parameter " << i << " address: " << params[i].get()
              << ", data: " << params[i]->get_data()
              << ", grad: " << params[i]->get_grad() << std::endl;
  }

  std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0),
                                           std::make_shared<Value>(2.0),
                                           std::make_shared<Value>(3.0)};

  auto output = mlp(x);
  std::cout << "\nOutput: " << output[0]->get_data() << std::endl;

  auto loss = output[0]->pow(std::make_shared<Value>(2));
  std::cout << "Loss: " << loss->get_data() << std::endl;

  loss->backward();

  std::cout << "\nParameters and gradients after backward pass:" << std::endl;
  for (size_t i = 0; i < params.size(); ++i) {
    std::cout << "Parameter " << i << " address: " << params[i].get()
              << ", data: " << params[i]->get_data()
              << ", grad: " << params[i]->get_grad() << std::endl;
  }

  return 0;
}
