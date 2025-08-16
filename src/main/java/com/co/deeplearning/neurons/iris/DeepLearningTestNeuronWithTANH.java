package com.co.deeplearning.neurons.iris;

import com.co.deeplearning.DeepLearningUtil;
import com.co.deeplearning.interfaces.IDeepLearningTest;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class DeepLearningTestNeuronWithTANH implements IDeepLearningTest {

    @Override
    public void execute() {
        // This method will contain the logic for the Iris neuron
        System.out.println("Executing Iris Neuron logic...");

        try (RecordReader reader = new CSVRecordReader(1, ',')) {
            // Initialize the RecordReader with the Iris dataset
            reader.initialize(DeepLearningUtil.getFileFromResource("iris.txt", reader));

            // Create a DataSetIterator from the RecordReader
            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    reader, 150, 4, 3
            );
            DataSet dataSet = iterator.next();
            dataSet.shuffle(42);
            // Iterate through the dataset and print the first record

            // Print the features and labels of the dataset
            //printDataSet(dataSet);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(dataSet);
            normalizer.transform(dataSet);

            // Print the normalized features and labels of the dataset
            //System.out.println("Normalized DataSet: ");
            //printDataSet(dataSet);

            SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.65);
            DataSet trainingData = splitTestAndTrain.getTrain();
            DataSet testData = splitTestAndTrain.getTest();

            MultiLayerConfiguration layerConfiguration = new NeuralNetConfiguration.Builder()
                    // activation functions: Hyperbolic tangent function (tanh)
                    .activation(Activation.TANH)
                    // weights initialization: Xavier initialization
                    .weightInit(WeightInit.XAVIER)
                    // learning rate: 0.1
                    .updater(new Adam(0.1))
                    // regularization: L2 regularization
                    .l2(0.0001)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                    .layer(2, new OutputLayer.Builder(
                            // Loss function NEGATIVELOGLIKELIHOOD used for multi-class classification problems
                            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            // Function SOFTMAX used in the output layer for multi-class classification problems
                            .activation(Activation.SOFTMAX)
                            // Output layer with 3 neurons (one for each class)
                            .nIn(3).nOut(3).build())
                    .backpropType(BackpropType.Standard)
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(layerConfiguration);
            model.init();
            System.out.println("Training model...");
            for (int i = 0; i < 1000; i++) {
                model.fit(trainingData);
                System.out.println("Epoch " + (i + 1) + " complete.");
            }
            System.out.println("Model training complete.");

            System.out.println("Evaluating model...");
            INDArray output = model.output(testData.getFeatures());
            Evaluation evaluation = new Evaluation();

            evaluation.eval(testData.getLabels(), output);
            System.out.println("Evaluation stats: ");
            System.out.println(evaluation.stats());


        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private  void printDataSet(DataSet dataSet) {
        // Print the features and labels of the dataset
        dataSet.asList()
                .forEach(dataSet1 -> System.out.println("Features: " + dataSet1.getFeatures()
                        + ", Labels: " + dataSet1.getLabels()));
    }
}
