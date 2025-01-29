# Simple Neural Network with Deeplearning4j

This repository contains a basic implementation of a neural network using Deeplearning4j (DL4J). The example demonstrates how to create, train, and make predictions with a simple feedforward neural network.

## Features

- Three-layer neural network architecture
- ReLU activation for hidden layers
- Softmax activation for output layer
- Adam optimizer
- Xavier weight initialization
- Example training with synthetic data
- Basic prediction functionality

## Prerequisites

- Java 8 or higher
- Maven or Gradle
- Basic understanding of neural networks

## Dependencies

Add these dependencies to your `pom.xml`:

```xml
<dependencies>
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-core</artifactId>
        <version>1.0.0-M2</version>
    </dependency>
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>1.0.0-M2</version>
    </dependency>
</dependencies>
```

## Network Architecture

1. Input Layer
   - Input features: 4

2. Hidden Layers
   - First hidden layer: 10 neurons with ReLU activation
   - Second hidden layer: 8 neurons with ReLU activation

3. Output Layer
   - Output classes: 3
   - Softmax activation
   - Negative log likelihood loss function

## Usage

1. Clone the repository:
```bash
https://github.com/ashishjsharda/dl4j-getting-started.git
```

2. Build the project:
```bash
mvn clean install
```

3. Run the example:
```bash
mvn exec:java -Dexec.mainClass="SimpleNeuralNetwork"
```

## Customization

You can modify the network architecture by adjusting these parameters in the code:

- Input features (`nIn` in first layer)
- Hidden layer sizes
- Number of output classes
- Learning rate (in Adam optimizer)
- Number of training epochs
- Batch size

## Example Output

The program will print:
- Training progress every 100 iterations
- Final test input and prediction results

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## Acknowledgments

- Based on the Deeplearning4j framework
- Inspired by basic neural network tutorials
- Created for educational purposes

## Note

This is a basic example intended for learning purposes. For production use, consider:
- Adding proper error handling
- Implementing cross-validation
- Adding model evaluation metrics
- Implementing proper data preprocessing
- Adding model saving/loading functionality
