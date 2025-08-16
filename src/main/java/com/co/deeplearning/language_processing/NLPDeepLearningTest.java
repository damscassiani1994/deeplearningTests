package com.co.deeplearning.language_processing;

import com.co.deeplearning.interfaces.IDeepLearningTest;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.io.ClassPathResource;
import java.io.IOException;
import java.util.Collection;

public class NLPDeepLearningTest implements IDeepLearningTest {

    @Override
    public void execute() {
        System.out.println("***Executing NLP Test logic...***");
        try {
            String filePath = new ClassPathResource("sentences.txt").getFile().getAbsolutePath();
            // Strip white spaces before and after for each line
            SentenceIterator iterator = new BasicLineIterator(filePath);
            TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
            /*
                CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
                So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
                Additionally it forces lower case for all tokens.
             */
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

            System.out.println("Building model...");
            Word2Vec word2Vec = new Word2Vec.Builder()
                    .minWordFrequency(1)
                    .iterations(1)
                    .layerSize(100)
                    .seed(42)
                    .windowSize(5)
                    .iterate(iterator)
                    .tokenizerFactory(tokenizerFactory)
                    .build();

            System.out.println("Fitting Word2Vec model...");
            word2Vec.fit();
            System.out.println("Model fitted successfully.");

            System.out.println("Writing word vectors to text file...");

            // Prints out the closest 10 words semantically to one word "For example, 'day'"
            String world = "vivía";
            Collection<String> list = word2Vec.wordsNearest(world, 10);
            System.out.println("10 Words closest to '"+ world +"': "  + list);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        System.out.println("***NLP Test completed.***");

    }
}
