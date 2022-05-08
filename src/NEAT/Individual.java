package NEAT;

import NEAT.Example.Classification;
import NEAT.NeuralNetwork.Connection;
import NEAT.NeuralNetwork.Node;
import NEAT.Example.Visual;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static java.awt.Color.*;
import static java.lang.String.valueOf;
import static NEAT.Example.Classification.*;
import static NEAT.Population.NUM_INDIVIDUALS;
import static NEAT.Example.Visual.*;

public class Individual implements Cloneable {
    private static int last_id = -1;
    private final int id;
    private int score;
    private int totX;
    private final Map<Node, Classification.Point> nodePositions = new HashMap<>();
    private int[] evaluations;
    private NeuralNetwork brain;
    private double fitness;


    public Individual() {
        this(++last_id);
    }
    public Individual(int id) {
        this.id = id;
    }
    public void init() {
        int startY = HEIGHT - 100;
        int endY = 100;
        int totY = startY - endY;
        int startX = 100;
        int endX = WIDTH - 100;
        totX = endX - startX;
        nodePositions.clear();
        int gapY = totY / (brain.layers()-1);
        int[] gapX = new int[brain.layers()];
        int[] drawnX = new int[brain.layers()];
        Map<Integer, List<Node>> nodes = this.brain.nodes().stream().collect(Collectors.groupingBy(Node::layer));
        Arrays.setAll(gapX, i -> totX / nodes.get(i).size());
        for (int i = 0; i < brain.nodes().size(); i++) {
            Node n = brain.nodes().get(i);
            int y = startY - (gapY * n.layer());
            int x = startX + (gapX[n.layer()] * drawnX[n.layer()]);
            x = n.layer() % 2 == 0 ? x : x + 50;
            nodePositions.put(n,new Classification.Point(x,y,-1));
            drawnX[n.layer()]++;
        }
    }
    public void evaluateAll() {
        this.evaluations = new int[NUM_POINTS];
        this.score = 0;
        for (int i = 0; i< NUM_POINTS; i++) {
            Classification.Point p = points[i];
            double[] out = this.brain.feedforward(new double[]{p.x(),p.y()});
            int res = out[0] >= 0 ? 1 : 0;
            this.evaluations[i] = res;
            if (res == p.label()) this.score ++;
        }
        this.fitness = score/(double)NUM_POINTS;
    }

    public Individual crossover(Individual parent) {
        Individual child = new Individual();
        child.brain = parent.fitness < this.fitness ?
                this.brain.crossover(parent.brain) :
                parent.brain.crossover(this.brain);
        child.brain.mutate();
        child.brain.generateNetwork();
        return child;
    }

    public void setBrain(NeuralNetwork brain) {
        this.brain = brain;
    }
    public NeuralNetwork getBrain() {
        return brain;
    }
    @Override public Individual clone() {
        try {
            return (Individual) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
    @Override public String toString() {
        return """
                Individual {
                    id=%d,
                    brain=%s,
                    fitness=%s,
                    score=%d
                }""".formatted(id, brain, fitness, score);
    }

    public void renderGenome(Graphics2D g) {
        g.setColor(new Color(0, 0, 0));
        g.fill3DRect(0, 0, Visual.WIDTH, Visual.HEIGHT, false);
        if (brain== null) return;

        g.setColor(Color.WHITE);

        nodePositions.forEach((n,point) -> {
            int x = (int) point.x(), y = (int) point.y();
            g.fill(new Ellipse2D.Double(x - 10, y - 10, 20, 20));
            g.drawString(valueOf(n.index()), x + 15, y);
            g.drawString(activationsNames[n.activationKey()], x + 30, y);
        });

        for (Connection c : brain.connections()) {
            if (c.enabled()){
                Classification.Point p1 = nodePositions.get(c.fromNode());
                Classification.Point p2 = nodePositions.get(c.toNode());
                double x1 = p1.x(), x2 = p2.x(), y1 = p1.y(), y2 = p2.y();
                g.draw(new Line2D.Double(x1,y1,x2,y2));
                g.drawString(String.format("%f",c.weight()),(int)((x1+x2) / 2) + 10, (int)(y1+y2) /2);
            }
        }

    }
    public void renderEvaluation(Graphics2D g) {
        for (int i = 0; i<NUM_POINTS; i++){
            Classification.Point p = points[i];
//            if (p == null) return;
            renderPointLabel(g, i, p);
        }
    }
    private void renderPointLabel(Graphics2D g, int i, Classification.Point p) {
        double eval = evaluations[i];
        if (p.label() == eval) g.setColor(eval == 0 ? BLUE : GREEN);
        else g.setColor(RED);
        g.fill(new Ellipse2D.Double(
                map(p.x(), 0, Math.PI*4, 0, WIDTH) - 4,
                p.y() * HEIGHT - 4 ,
                8,
                8
        ));
    }

    public double fitness() {
        return this.fitness;
    }
}
