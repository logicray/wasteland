package person.me.galaxy.ml.word2vec.domain;

public class HiddenNeuron extends Neuron{
    
    public double[] syn1 ; //hidden->out
    
    public HiddenNeuron(int layerSize){
        syn1 = new double[layerSize] ;
    }
    
}
