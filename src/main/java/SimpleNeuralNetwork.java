import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SimpleNeuralNetwork {
    public static void main(String[] args) {
        // Define the neural network architecture
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.01))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4)  // Input features
                        .nOut(10) // Hidden layer size
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(8)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(8)
                        .nOut(3)  // Number of classes
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // Create and initialize the neural network
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Create sample training data
        int numSamples = 150;
        INDArray features = Nd4j.rand(numSamples, 4);
        INDArray labels = Nd4j.zeros(numSamples, 3);
        for (int i = 0; i < numSamples; i++) {
            int labelIdx = i % 3;
            labels.putScalar(new int[]{i, labelIdx}, 1.0);
        }

        // Create DataSet
        DataSet trainingData = new DataSet(features, labels);

        // Train the model
        int numEpochs = 1000;
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
        }

        // Example prediction
        INDArray testInput = Nd4j.rand(1, 4);
        INDArray output = model.output(testInput);
        System.out.println("Test Input: " + testInput);
        System.out.println("Predicted Output: " + output);
    }
}