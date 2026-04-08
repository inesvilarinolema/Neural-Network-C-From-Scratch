# Multilayer Perceptron from Scratch in C (XOR Problem) 

## About The Project
This project implements a **Multi-Layer Neural Network from scratch in C** to solve the non-linearly separable XOR problem. 

Instead of relying on high-level Python libraries, this project demonstrates a deep, mathematical understanding of Artificial Neural Networks by manually implementing the **Error Back-Propagation algorithm**, weight initialization, and activation functions in a low-level language.

## Key Features
* **Custom Backpropagation:** Manual calculation of output errors, hidden layer propagation, and weight updates.
* **Math Implementation:** Custom Sigmoid activation function and its derivative.
* **Dynamic Convergence:** The network trains dynamically until the Mean Squared Error (MSE) falls below a strict threshold ($E_{max} = 0.001$).

## Architecture
* **Input Layer:** 2 neurons + 1 Bias ($x_3 = -1$)
* **Hidden Layer:** 2 neurons + 1 Bias ($y_3 = -1$)
* **Output Layer:** 1 neuron
* **Activation Function:** Unipolar Sigmoid
* **Learning Rate ($\eta$):** 0.5

## How to Compile and Run
Since the project uses the math library (`<math.h>`), you must link it during compilation using `-lm`.

```bash
# 1. Compile the code
gcc XOR.c -o xor_nn -lm

# 2. Run the executable
./xor_nn
```

## 📊 Results & Verification
The network initializes with random weights to break symmetry and typically converges around 6,000 - 10,000 epochs.

**Output Example:**
```text
Epoch: 1000, Error: 0.124500
Epoch: 2000, Error: 0.089300
...
---CONVERGENCE REACHED---
Epoch: 6918, Final Error: 0.000998

---VERIFICATION---
X1  X2  | Target | Output
0   0   |    0   | 0.0210
0   1   |    1   | 0.9654
1   0   |    1   | 0.9632
1   1   |    0   | 0.0315
```
*The outputs are extremely close to the binary targets, confirming the network successfully learned the non-linear decision boundary.*
