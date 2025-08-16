package com.co.deeplearning;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.StreamSupport;

public final class DeepLearningUtil {

    public static FileSplit getFileFromResource(String name, RecordReader reader) throws IOException {
        return new FileSplit(new ClassPathResource(name).getFile());
    }

    public static long getFileSize(RecordReader reader) throws IOException {
        Iterator<List<Writable>> iterator = new Iterator<List<Writable>>() {
            @Override
            public boolean hasNext() {
                return reader.hasNext();
            }

            @Override
            public List<Writable> next() {
                return reader.next();
            }
        } ;

        long count = StreamSupport.stream(Spliterators.spliteratorUnknownSize(iterator, Spliterator.ORDERED), false)
                .count();
        reader.reset();
        return count;
    }


    public static MultiLayerNetwork trainModel(MultiLayerConfiguration layerConfiguration, DataSet trainingData, int epochs) {
        MultiLayerNetwork model = new MultiLayerNetwork(layerConfiguration);
        model.init();
        for (int i = 0; i < epochs; i++) {
            model.fit(trainingData);
            System.out.println("Epoch " + (i + 1) + " complete.");
        }
        return  model;
    }

}
