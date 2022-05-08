package NEAT.Example;

import NEAT.Individual;

import javax.swing.*;
import java.awt.*;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class Visual {
    public static final int WIDTH = 700, HEIGHT = 700;
    public static final Dimension screen = new Dimension(WIDTH, HEIGHT);
    private static final Classification classification = new Classification();
    public static final String[] activationsNames = {"Sigmoid","Identity","Step","Tanh","ReLu","Sin","Cos"};

    private final JFrame annFrame = new JFrame();
    private final JFrame classificationFrame = new JFrame();
    private final JPanel annScene = new JPanel();
    private final JPanel classificationScene = new JPanel();
    private Instant now = Instant.now();


    {
       initFrame(annFrame,annScene,0);
       initFrame(classificationFrame,classificationScene, WIDTH + 50);
    }

    public static void initFrame(JFrame frame,JPanel scene, int dx) {
        frame.setSize(screen);
        frame.setLocation(dx,50);
        scene.setPreferredSize(screen);
        frame.add(scene);
        frame.pack();
        frame.setVisible(true);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    public static void main(String[] args) {
        new Visual().start();
    }
    public void start() {
        ScheduledExecutorService service = Executors.newSingleThreadScheduledExecutor();
        final Individual[] best = new Individual[1];
        service.scheduleAtFixedRate(() -> {
            annFrame.setTitle("Generation " + classification.getPopulation().generation +
                    " -- "+ classification.getPopulation().bestFitness + " best fit."+
                    " -- "+ Duration.between(now,now = Instant.now()).getNano() + " ns");
            classification.getPopulation().initPopulation();
            classification.initPoints();
            classification.evaluateAll();
            if (classification.getPopulation().bestPlayer() != best[0]){
                best[0] = classification.getPopulation().bestPlayer();
                best[0].renderGenome((Graphics2D) annScene.getGraphics());
                classification.renderPoints((Graphics2D) classificationScene.getGraphics());
                best[0].renderEvaluation((Graphics2D) classificationScene.getGraphics());
            }

            classification.getPopulation().naturalSelection();
        }, 500, 120, TimeUnit.MILLISECONDS);
    }


}
