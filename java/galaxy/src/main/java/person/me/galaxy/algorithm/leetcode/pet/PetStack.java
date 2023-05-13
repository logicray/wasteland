package person.me.galaxy.algorithm.leetcode.pet;

import java.util.ArrayList;
import java.util.List;

public class PetStack {
    private List<Pet> petStack;

    public PetStack() {
        this.petStack = new ArrayList<>();
    }

    public void add(Pet pet){
        petStack.add(pet);
    }

    public void pollAll(){
        for (int i=0;i<petStack.size();i++){
           Pet pet = petStack.remove(i);
            System.out.println(pet);
        }
    }

    public void pollCat(){

    }

    public void pollDog(){

    }

    public static void main(String[] args) {
        PetStack stack = new PetStack();
        stack.add(new Cat());
        stack.add(new Dog());

    }
}
