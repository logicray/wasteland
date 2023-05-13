package person.me.galaxy.ml.word2vec.domain;

import java.util.*;

public class WordNeuron extends Neuron {
  private String name;
  public double[] syn0 = null; // input->hidden
  public List<Neuron> neuronPathList = null;// 路径神经元
  public int[] codeArr = null;

  public WordNeuron(String name, double freq, int layerSize) {
    this.name = name;
    this.freq = freq;
    this.syn0 = new double[layerSize];
    Random random = new Random();
    for (int i = 0; i < syn0.length; i++) {
      syn0[i] = (random.nextDouble() - 0.5) / layerSize;
    }
  }

  /**
   * 用于有监督的创造hoffman tree
   * 
   * @param name
   * @param freq
   * @param layerSize
   */
  public WordNeuron(String name, double freq, int category, int layerSize) {
    this.name = name;
    this.freq = freq;
    this.syn0 = new double[layerSize];
    this.category = category;
    Random random = new Random();
    for (int i = 0; i < syn0.length; i++) {
      syn0[i] = (random.nextDouble() - 0.5) / layerSize;
    }
  }


  public List<Neuron> makeNeurons() {
    if (neuronPathList != null) {
      return neuronPathList;
    }
    Neuron neuron = this;
    neuronPathList = new LinkedList<>();
    while ((neuron = neuron.parent) != null) {
      neuronPathList.add(neuron);
    }
    Collections.reverse(neuronPathList);
    codeArr = new int[neuronPathList.size()];

    for (int i = 1; i < neuronPathList.size(); i++) {
      codeArr[i - 1] = neuronPathList.get(i).code;
    }
    codeArr[codeArr.length - 1] = this.code;

    return neuronPathList;
  }

  @Override
  public String toString() {
    return "WordNeuron{" +
            "name='" + name + '\'' +
            ", syn0=" + Arrays.toString(syn0) +
            ", neuronPathList=" + neuronPathList +
            ", codeArr=" + Arrays.toString(codeArr) +
            ", freq=" + freq +
            ", parent=" + parent +
            ", code=" + code +
            ", category=" + category +
            '}';
  }
}