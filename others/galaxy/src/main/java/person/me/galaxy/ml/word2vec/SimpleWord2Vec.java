package person.me.galaxy.ml.word2vec;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * a very simple word embedding, without negative sampling or Hierarchical softmax,
 * just three layer neural network.
 *
 * 输入的原始数据需为每行一句话，每个词之间使用 \t 分割，
 * 整体分为预处理和训练两步
 */
public class SimpleWord2Vec {
    //    private double sample = 1e-3;
//    private double alpha = 0.025;
//    private double startingAlpha = alpha;
//    private int trainWordsCount = 0;

    public SimpleWord2Vec(String filePath) {
        File file = new File(filePath);

    }

    public void process(){

    }

    public void train(){

    }

//    private void trainModel(File file) throws IOException {
//        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
//            String temp = null;
//            long nextRandom = 5;
//            int wordCount = 0;
//            int lastWordCount = 0;
//            int wordCountActual = 0;
//            // 一次加载一行
//            while ((temp = br.readLine()) != null) {
//                if (wordCount - lastWordCount > 10000) {
//                    System.out.println("alpha:" + alpha + "\tProgress: "
//                            + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100) + "%");
//                    wordCountActual += (wordCount - lastWordCount);
//                    lastWordCount = wordCount;
//                    alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
//                    if (alpha < startingAlpha * 0.0001) {
//                        alpha = startingAlpha * 0.0001;
//                    }
//                }
//                String[] strs = temp.split("\t");
//                wordCount += strs.length;
//                List<WordNeuron> sentence = new ArrayList<>();
//                for (String str : strs) {
//                    Neuron entry = wordMap.get(str);
//                    if (entry == null) {
//                        continue;
//                    }
//                    // The subsampling randomly discards frequent words while keeping the
//                    // ranking same
//                    if (sample > 0) {
//                        double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
//                                * (sample * trainWordsCount) / entry.freq;
//                        nextRandom = nextRandom * 25214903917L + 11;
//                        if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
//                            continue;
//                        }
//                    }
//                    sentence.add((WordNeuron) entry);
//                }
//
//                for (int index = 0; index < sentence.size(); index++) {
//                    nextRandom = nextRandom * 25214903917L + 11;
//                    if (isCBOW) {
//                        cbowGram(index, sentence, (int) nextRandom % WINDOW);
//                    } else {
//                        skipGram(index, sentence, (int) nextRandom % WINDOW);
//                    }
//                }
//
//            }
//            System.out.println("Vocab size: " + wordMap.size());
//            System.out.println("Words in train file: " + trainWordsCount);
//            System.out.println("success train over!");
//        }
//    }

    public static void main(String[] args) {
        SimpleWord2Vec word2Vec = new SimpleWord2Vec("./");
        word2Vec.process();
        word2Vec.train();
//        word2Vec.distance();
    }
}
