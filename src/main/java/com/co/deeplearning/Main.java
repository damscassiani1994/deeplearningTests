package com.co.deeplearning;


import com.co.deeplearning.fraudulent_transactions.FraudulentTransaction;
import com.co.deeplearning.interfaces.IDeepLearningTest;
import com.co.deeplearning.language_processing.NLPDeepLearningTest;
import com.co.deeplearning.neurons.iris.DeepLearningTestNeuronWithRELU;
import com.co.deeplearning.neurons.iris.DeepLearningTestNeuronWithTANH;

public class Main {
    public static void main(String[] args) {
        IDeepLearningTest test = getIrisNeuron("FRAUD");
        test.execute();
    }


    private static IDeepLearningTest getIrisNeuron(String type) {
        return switch (type) {
            case "TANH" -> new DeepLearningTestNeuronWithTANH();
            case "RELU" -> new DeepLearningTestNeuronWithRELU(); // Uncomment when implemented
            case "NLP" -> new NLPDeepLearningTest();
            case "FRAUD" -> new FraudulentTransaction();
            default -> throw new IllegalArgumentException("Unknown neuron type: " + type);
        };
    }
}