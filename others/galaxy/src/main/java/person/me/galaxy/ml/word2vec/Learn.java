package person.me.galaxy.ml.word2vec;


import person.me.galaxy.ml.word2vec.domain.HiddenNeuron;
import person.me.galaxy.ml.word2vec.domain.Neuron;
import person.me.galaxy.ml.word2vec.domain.WordNeuron;
import person.me.galaxy.ml.word2vec.util.Huffman;
import person.me.galaxy.ml.word2vec.util.MapCount;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 */
public class Learn {

  private Map<String, Neuron> wordMap = new HashMap<>();
  //每个词的特征向量维数
  private int LAYER_SIZE = 200;
   //上下文窗口大小
  private int WINDOW = 5;

  private double sample = 1e-3;
  private double alpha = 0.025;
  private double startingAlpha = alpha;

  public int EXP_TABLE_SIZE = 1000;

  private Boolean isCBOW = false;

  public double[] expTable = new double[EXP_TABLE_SIZE];

  private int trainWordsCount = 0;

  private int MAX_EXP = 6;

  public Learn(Boolean isCbow, Integer layerSize, Integer window, Double alpha, Double sample) {
    createExpTable();
    if (isCbow != null) {
      this.isCBOW = isCbow;
    }
    if (layerSize != null)
      this.LAYER_SIZE = layerSize;
    if (window != null)
      this.WINDOW = window;
    if (alpha != null)
      this.alpha = alpha;
    if (sample != null)
      this.sample = sample;
  }

  public Learn() {
    createExpTable();
  }

  /**
   * trainModel
   * 
   * @throws IOException
   */
  private void trainModel(File file) throws IOException {
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
      String temp = null;
      long nextRandom = 5;
      int wordCount = 0;
      int lastWordCount = 0;
      int wordCountActual = 0;
      // 一次加载一行
      while ((temp = br.readLine()) != null) {
        if (wordCount - lastWordCount > 10000) {
          System.out.println("alpha:" + alpha + "\tProgress: "
              + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100) + "%");
          wordCountActual += (wordCount - lastWordCount);
          lastWordCount = wordCount;
          alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
          if (alpha < startingAlpha * 0.0001) {
            alpha = startingAlpha * 0.0001;
          }
        }
        String[] strs = temp.split("\t");
        wordCount += strs.length;
        List<WordNeuron> sentence = new ArrayList<>();
        for (String str : strs) {
          Neuron entry = wordMap.get(str);
          if (entry == null) {
            continue;
          }
          // The subsampling randomly discards frequent words while keeping the
          // ranking same
          if (sample > 0) {
            double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                    * (sample * trainWordsCount) / entry.freq;
            nextRandom = nextRandom * 25214903917L + 11;
            if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
              continue;
            }
          }
          sentence.add((WordNeuron) entry);
        }

        for (int index = 0; index < sentence.size(); index++) {
          nextRandom = nextRandom * 25214903917L + 11;
          if (isCBOW) {
            cbowGram(index, sentence, (int) nextRandom % WINDOW);
          } else {
            skipGram(index, sentence, (int) nextRandom % WINDOW);
          }
        }

      }
      System.out.println("Vocab size: " + wordMap.size());
      System.out.println("Words in train file: " + trainWordsCount);
      System.out.println("success train over!");
    }
  }

  /**
   * skip gram 模型训练
   * 
   * @param sentence
   * @param b
   */
  private void skipGram(int index, List<WordNeuron> sentence, int b) {
    WordNeuron word = sentence.get(index);
    int a, c = 0;
    for (a = b; a < WINDOW * 2 + 1 - b; a++) {
      if (a == WINDOW) {
        continue;
      }
      //if set index-window as start, index+window as end, actually, c is from start+b to end-b
      c = index - WINDOW + a;
      if (c < 0 || c >= sentence.size()) {
        continue;
      }

      double[] neu1e = new double[LAYER_SIZE];// 误差项
      // HIERARCHICAL SOFTMAX
      List<Neuron> neurons = word.neuronPathList;
      WordNeuron we = sentence.get(c);
      for (int i = 0; i < neurons.size(); i++) {
        HiddenNeuron out = (HiddenNeuron) neurons.get(i);
        double f = 0;
        // Propagate hidden -> output
        for (int j = 0; j < LAYER_SIZE; j++) {
          f += we.syn0[j] * out.syn1[j];
        }
        if (f <= -MAX_EXP || f >= MAX_EXP) {
          continue;
        } else {
          f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
          f = expTable[(int) f];
        }
        // 'g' is the gradient multiplied by the learning rate
        double g = (1 - word.codeArr[i] - f) * alpha;
        // Propagate errors output -> hidden
        for (c = 0; c < LAYER_SIZE; c++) {
          neu1e[c] += g * out.syn1[c];
        }
        // Learn weights hidden -> output
        for (c = 0; c < LAYER_SIZE; c++) {
          out.syn1[c] += g * we.syn0[c];
        }
      }

      // Learn weights input -> hidden
      for (int j = 0; j < LAYER_SIZE; j++) {
        we.syn0[j] += neu1e[j];
      }
    }

  }

  /**
   * 词袋模型
   * 
   * @param index
   * @param sentence
   * @param b
   */
  private void cbowGram(int index, List<WordNeuron> sentence, int b) {
    WordNeuron word = sentence.get(index);
    int a, c = 0;

    List<Neuron> neurons = word.neuronPathList;
    double[] neu1e = new double[LAYER_SIZE];// 误差项
    double[] neu1 = new double[LAYER_SIZE];// 误差项
    WordNeuron last_word;

    for (a = b; a < WINDOW * 2 + 1 - b; a++)
      if (a != WINDOW) {
        c = index - WINDOW + a;
        if (c < 0 || c >= sentence.size())
          continue;
        last_word = sentence.get(c);
        if (last_word == null)
          continue;
        for (c = 0; c < LAYER_SIZE; c++)
          neu1[c] += last_word.syn0[c];
      }

    // HIERARCHICAL SOFTMAX
    for (int d = 0; d < neurons.size(); d++) {
      HiddenNeuron out = (HiddenNeuron) neurons.get(d);
      double f = 0;
      // Propagate hidden -> output
      for (c = 0; c < LAYER_SIZE; c++)
        f += neu1[c] * out.syn1[c];
      if (f <= -MAX_EXP)
        continue;
      else if (f >= MAX_EXP)
        continue;
      else
        f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
      // 'g' is the gradient multiplied by the learning rate
      // double g = (1 - word.codeArr[d] - f) * alpha;
      // double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
      double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
      //
      for (c = 0; c < LAYER_SIZE; c++) {
        neu1e[c] += g * out.syn1[c];
      }
      // Learn weights hidden -> output
      for (c = 0; c < LAYER_SIZE; c++) {
        out.syn1[c] += g * neu1[c];
      }
    }
    for (a = b; a < WINDOW * 2 + 1 - b; a++) {
      if (a != WINDOW) {
        c = index - WINDOW + a;
        if (c < 0)
          continue;
        if (c >= sentence.size())
          continue;
        last_word = sentence.get(c);
        if (last_word == null)
          continue;
        for (c = 0; c < LAYER_SIZE; c++)
          last_word.syn0[c] += neu1e[c];
      }

    }
  }

  /**
   * 统计词频
   * 
   * @param file
   * @throws IOException
   */
  private Map<String, Neuron> readVocab(File file) throws IOException {
    ///m每个词出现次数和总次数统计，，
    MapCount<String> mc = new MapCount<>();
    try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
      String temp = null;
      while ((temp = br.readLine()) != null) {
        String[] split = temp.split("\t");
        trainWordsCount += split.length;
        for (String string : split) {
          mc.add(string);
        }
      }
    }

    //词频
    Map<String, Neuron> wordMap =  new HashMap<>();
    for (Entry<String, Integer> element : mc.get().entrySet()) {
      WordNeuron wordNeuron = new WordNeuron(element.getKey(), (double) element.getValue() / mc.size(), LAYER_SIZE);
      wordMap.put(element.getKey(), wordNeuron);
    }
    return wordMap;
  }

  /**
   * 对文本进行预分类
   * 
   * @param files
   * @throws IOException
   * @throws FileNotFoundException
   */
  private void readVocabWithSupervised(File[] files) throws IOException {
    for (int category = 0; category < files.length; category++) {
      // 对多个文件学习
      MapCount<String> mc = new MapCount<>();
      try (BufferedReader br = new BufferedReader(new InputStreamReader(
          new FileInputStream(files[category])))) {
        String temp = null;
        while ((temp = br.readLine()) != null) {
          String[] split = temp.split(" ");
          trainWordsCount += split.length;
          for (String string : split) {
            mc.add(string);
          }
        }
      }
      for (Entry<String, Integer> element : mc.get().entrySet()) {
        double tarFreq = (double) element.getValue() / mc.size();
        if (wordMap.get(element.getKey()) != null) {
          double srcFreq = wordMap.get(element.getKey()).freq;
          if (srcFreq >= tarFreq) {
            continue;
          } else {
            Neuron wordNeuron = wordMap.get(element.getKey());
            wordNeuron.category = category;
            wordNeuron.freq = tarFreq;
          }
        } else {
          wordMap.put(element.getKey(), new WordNeuron(element.getKey(),
              tarFreq, category, LAYER_SIZE));
        }
      }
    }
  }

  /**
   * pre compute the exp() table f(x) = x / (x + 1)
   * logistic 函数查询表
   * EXP_TABLE_SIZE 越大，精度越高
   */
  public void createExpTable() {
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
      expTable[i] = Math.exp((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
      expTable[i] = expTable[i] / (expTable[i] + 1);
    }
  }

  /**
   * 根据文件学习
   * 
   * @param file
   * @throws IOException
   */
  public void learnFile(File file) throws IOException {
    System.out.println("before");
    System.out.println("wordMap.size():" + wordMap.size());

    //读入数据，统计词频
    wordMap = readVocab(file);
    String key = wordMap.keySet().toArray(new String[0])[15];
    System.out.println(key);
    System.out.println(wordMap.get(key).toString());
    System.out.println(wordMap.get(key).toString());

    // 根据词频构建Huffman树
    Huffman huffman = new Huffman(LAYER_SIZE);
    huffman.make(wordMap.values());

    System.out.println(wordMap.get(key).toString());

    //为每个Neuron 建立从该节点到根节点的路径，以及对应的code列表
    for (Neuron neuron : wordMap.values()) {
      ((WordNeuron) neuron).makeNeurons();
    }
    System.out.println(wordMap.get(key).toString());

    trainModel(file);
  }

  /**
   * 根据预分类的文件学习
   * 
   * @param summaryFile
   *          合并文件
   * @param classifiedFiles
   *          分类文件
   * @throws IOException
   */
  public void learnFile(File summaryFile, File[] classifiedFiles) throws IOException {
    readVocabWithSupervised(classifiedFiles);
    new Huffman(LAYER_SIZE).make(wordMap.values());
    // 查找每个神经元
    for (Neuron neuron : wordMap.values()) {
      ((WordNeuron) neuron).makeNeurons();
    }
    trainModel(summaryFile);
  }

  /**
   * 保存模型
   */
  public void saveModel(File file) {
    // TODO Auto-generated method stub

    try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
      dataOutputStream.writeInt(wordMap.size());
      dataOutputStream.writeInt(LAYER_SIZE);
      double[] syn0 = null;
      for (Entry<String, Neuron> element : wordMap.entrySet()) {
        dataOutputStream.writeUTF(element.getKey());
        syn0 = ((WordNeuron) element.getValue()).syn0;
        for (double d : syn0) {
          dataOutputStream.writeFloat(((Double) d).floatValue());
        }
      }
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

}
