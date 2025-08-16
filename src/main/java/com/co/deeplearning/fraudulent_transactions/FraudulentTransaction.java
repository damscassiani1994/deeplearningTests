package com.co.deeplearning.fraudulent_transactions;

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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

public class FraudulentTransaction implements IDeepLearningTest {


    @Override
    public void execute() {
        System.out.println("Executing Fraudulent Transaction Detection Test");
        // Implementation for fraudulent transaction detection goes here
        try (RecordReader reader = new CSVRecordReader(1, ',')){
            FileSplit file = DeepLearningUtil.getFileFromResource("datasets/creditcard.csv", reader);
            // Initialize the RecordReader with the fraudulent transactions dataset 284807
            reader.initialize(file);
            int batchSize = Math.toIntExact(DeepLearningUtil.getFileSize(reader)) - 1;
            System.out.println("Total records in the dataset: "+ batchSize);
            int labelInex = 30; // Index of the label column (1 for 'Class' in the dataset)
            int numClasses = 2; // Number of classes (0 and 1 for 'Class' in the dataset)

            // Create a DataSetIterator from the RecordReader
            DataSetIterator iterator = new RecordReaderDataSetIterator(reader, batchSize, labelInex, numClasses);

            DataSet dataSet  = iterator.next();
            dataSet.shuffle(42);

            System.out.println("Dataset before normalization: " + dataSet);

            DataNormalization dataNormalization = new NormalizerStandardize();
            dataNormalization.fit(dataSet);
            dataNormalization.transform(dataSet);

            System.out.println("Dataset after normalization: " + dataSet);

            // Divide the dataset into training and testing sets
            SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.65);
            DataSet trainingData = splitTestAndTrain.getTrain();
            DataSet testingData = splitTestAndTrain.getTest();

           MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration
                    .Builder()
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.1))
                    .l2(1e-5)
                    .list()
                    .layer(0, new DenseLayer
                            .Builder()
                            .nIn(labelInex)
                            .nOut(15).build())
                    .layer(1, new DenseLayer
                            .Builder()
                            .nIn(15)
                            .nOut(10).build())
                   .layer(2, new DenseLayer
                            .Builder()
                            .nIn(10)
                            .nOut(5).build())
                   .layer(3, new OutputLayer
                            .Builder()
                           .nIn(5)
                           .nOut(2).build()
                   ).backpropType(BackpropType.Standard)
                     .build();
            System.out.println("Training model...");
            MultiLayerNetwork model = DeepLearningUtil.trainModel(multiLayerConfiguration,
                    trainingData, 1000);
            System.out.println("Model training complete.");

            System.out.println("Evaluating model...");
            INDArray output = model.output(testingData.getFeatures());
            Evaluation evaluation = new Evaluation();

            evaluation.eval(testingData.getLabels(), output);
            System.out.println("Evaluation stats: ");
            System.out.println(evaluation.stats());


            System.out.println("Finished executing Fraudulent Transaction Detection Test");
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
