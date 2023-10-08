#include <Arduino.h>

// Define the neural network parameters
const int inputNodes = 3;       // Number of input nodes (features)
const int hiddenNodes = 6;      // Number of nodes in the hidden layer
const int outputNodes = 1;      // Number of output nodes
float temperatureMin = 0.0;
float temperatureMax = 100.0;

float humidityMin = 0.0;
float humidityMax = 100.0;

float windSpeedMin = 0.0;
float windSpeedMax = 50.0;
// Define the neural network weights
float hiddenLayerWeights[inputNodes][hiddenNodes];
float outputLayerWeights[hiddenNodes][outputNodes];

// Define the neural network biases
float hiddenLayerBiases[hiddenNodes];
float outputLayerBiases[outputNodes];

// Define learning rate
const float learningRate = 0.1;    // You can adjust this as per your requirement

// Activation function (sigmoid)
float sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

// Derivative of the activation function
float sigmoidDerivative(float x) {
  float sigmoidX = sigmoid(x);
  return sigmoidX * (1 - sigmoidX);
}

// Forward propagation
float feedForward(float inputs[inputNodes]) {
  // Calculate outputs of the hidden layer
  float hiddenLayerOutputs[hiddenNodes];
  for (int i = 0; i < hiddenNodes; i++) {
    float weightedSum = 0;
    for (int j = 0; j < inputNodes; j++) {
      weightedSum += inputs[j] * hiddenLayerWeights[j][i];
    }
    hiddenLayerOutputs[i] = sigmoid(weightedSum + hiddenLayerBiases[i]);
  }

  // Calculate output of the neural network
  float output = 0;
  for (int i = 0; i < outputNodes; i++) {
    float weightedSum = 0;
    for (int j = 0; j < hiddenNodes; j++) {
      weightedSum += hiddenLayerOutputs[j] * outputLayerWeights[j][i];
    }
    output = sigmoid(weightedSum + outputLayerBiases[i]);
  }

  return output;
}

// Backpropagation
void backpropagation(float inputs[inputNodes], float target) {
  // Perform forward propagation to get outputs of each layer
  float hiddenLayerOutputs[hiddenNodes];
  for (int i = 0; i < hiddenNodes; i++) {
    float weightedSum = 0;
    for (int j = 0; j < inputNodes; j++) {
      weightedSum += inputs[j] * hiddenLayerWeights[j][i];
    }
    hiddenLayerOutputs[i] = sigmoid(weightedSum + hiddenLayerBiases[i]);
  }
  
  float output = 0;
  for (int i = 0; i < outputNodes; i++) {
    float weightedSum = 0;
    for (int j = 0; j < hiddenNodes; j++) {
      weightedSum += hiddenLayerOutputs[j] * outputLayerWeights[j][i];
    }
    output = sigmoid(weightedSum + outputLayerBiases[i]);
  }

  // Calculate the loss
  float error = target - output;
  // Calculate the derivative of the activation function
  float outputDerivative = sigmoidDerivative(output);
  // Calculate the output layer gradients
  float outputGradients[outputNodes];
  for (int i = 0; i < outputNodes; i++) {
    outputGradients[i] = error * outputDerivative;
  }

  // Adjust the weights and biases of the output layer
  for (int i = 0; i < hiddenNodes; i++) {
    for (int j = 0; j < outputNodes; j++) {
      outputLayerWeights[i][j] += learningRate * hiddenLayerOutputs[i] * outputGradients[j];
    }
  }
  for (int i = 0; i < outputNodes; i++) {
    outputLayerBiases[i] += learningRate * outputGradients[i];
  }

  // Calculate the hidden layer gradients
  float hiddenGradients[hiddenNodes];
  for (int i = 0; i < hiddenNodes; i++) {
    float weightedSum = 0;
    for (int j = 0; j < outputNodes; j++) {
      weightedSum += outputGradients[j] * outputLayerWeights[i][j];
    }
    hiddenGradients[i] = weightedSum * sigmoidDerivative(hiddenLayerOutputs[i]);
  }

  // Adjust the weights and biases of the hidden layer
  for (int i = 0; i < inputNodes; i++) {
    for (int j = 0; j < hiddenNodes; j++) {
      hiddenLayerWeights[i][j] += learningRate * inputs[i] * hiddenGradients[j];
    }
  }
  for (int i = 0; i < hiddenNodes; i++) {
    hiddenLayerBiases[i] += learningRate * hiddenGradients[i];
  }
}

void setup() {
  // Initialize the weights and biases (you can modify them as per your training)
  // ...
Serial.begin(115200);
  // Initialize other necessary setup for the ESP32
  // ...
}

void loop() {
  // Read inputs from sensors
  float temperature = 25;
  float humidity = 80;
  float windSpeed = 13;

  temperature = (temperature - temperatureMin) / (temperatureMax - temperatureMin);
  humidity = (humidity - humidityMin) / (humidityMax - humidityMin);
  windSpeed = (windSpeed - windSpeedMin) / (windSpeedMax - windSpeedMin);

  // Create an input array
  float inputs[inputNodes];
  inputs[0] = temperature;
  inputs[1] = humidity;
  inputs[2] = windSpeed;

  // Perform forward propagation
  float predictedWeather = feedForward(inputs);

  // Perform backpropagation with a target value
  float target = 0.75; // Set the target value (you can modify it as per your training data)
  backpropagation(inputs, target);
 Serial.print("Temperature: ");
  Serial.println(temperature);
  Serial.print("Humidity: ");
  Serial.println(humidity);
  Serial.print("Wind Speed: ");
  Serial.println(windSpeed);
  Serial.print("Predicted Weather: ");
  Serial.println(predictedWeather);
  // Do something with the predicted weather value
  // ...

  delay(1000);  // Delay for a second before predicting again
}
