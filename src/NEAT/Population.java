package NEAT;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static NEAT.Example.Settings.RANDOM;

public class Population {
    public static final int NUM_INDIVIDUALS = 500;
    private final Individual[] population;
    private Individual bestPlayer;
    public double bestFitness;
    private final List<Integer> matingPool;
    public int generation;
    public Population(int numInputs, int numOutputs) {
        this.generation = 0;
        this.bestFitness = 0;
        this.population = new Individual[NUM_INDIVIDUALS];
        for (int i = 0; i< NUM_INDIVIDUALS; i++) {
            Individual individual = new Individual();
            individual.setBrain(NeuralNetwork.createNetwork(numInputs,numOutputs));
            this.population[i] = individual;
        }
        this.matingPool = new ArrayList<>();
    }
    public void initPopulation() {
        for (int i = 0; i< NUM_INDIVIDUALS; i++) {
            this.population[i].init();
        }
    }
    public void fillMatingPool() {
        this.matingPool.clear();
        for (int i = 0;i< this.population.length; i++) {
            if (this.population[i].fitness() > this.bestFitness){
                this.bestFitness = this.population[i].fitness();
                this.bestPlayer = this.population[i];
            }
            double n = this.population[i].fitness() * 100;
            for (int k = 0; k< n; k++) {
                this.matingPool.add(i);
            }
        }
    }

    public void naturalSelection() {
        Individual[] children = new Individual[this.population.length];
        this.fillMatingPool();
        for (int i = 0; i< this.population.length; i++){
            Individual p1 = this.selectPlayer();
            Individual p2 = this.selectPlayer();
            children[i] = (p1.fitness() > p2.fitness() ? p1.crossover(p2) : p2.crossover(p1));
        }

        for (int i = 0; i< this.population.length; i++){
            this.population[i] = children[i];
            this.population[i].getBrain().generateNetwork();
        }

        this.generation++;
    }

    private Individual selectPlayer() {
        if (RANDOM.nextDouble() < 0.1) {
            return this.bestPlayer;
        }
        return this.population[this.matingPool.get(RANDOM.nextInt(this.matingPool.size()))];
    }

    public Individual bestPlayer() {
        return bestPlayer == null ? Arrays.stream(this.population).
                max(Comparator.comparingDouble(Individual::fitness)).
                orElse(this.population[0]) : bestPlayer;
    }

    public Individual[] players() {
        return this.population;
    }
}
