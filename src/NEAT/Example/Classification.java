package NEAT.Example;
import NEAT.Individual;
import NEAT.Population;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.util.Arrays;
import java.util.function.DoubleFunction;

import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static NEAT.Example.Visual.HEIGHT;
import static NEAT.Example.Visual.WIDTH;

public class Classification {
    public static final int NUM_POINTS = 1000;
    public static final Point[] points = new Point[NUM_POINTS];
    private final Population population;
    public static DoubleFunction<Double> myFunction = x -> .4 * sin(x) * .3 * cos(x) + 0.5;
    private boolean ready = false;
    public Classification() {
        this.population = new Population(2,1);
    }
    public Population getPopulation() {
        return population;
    }
    public void initPoints() {
        if (ready) return;
        for (int i = 0; i< NUM_POINTS;i++) {
            double y = Math.random(), x = Math.random() * Math.PI * 4;
            int type = y > myFunction.apply(x) ? 1 : 0;
            points[i] = new Point(x, y,type);
        }
        ready = true;
    }
    public void renderPoints(Graphics2D g) {
        g.setColor(new Color(0, 0, 0));
        g.fill3DRect(0, 0, Visual.WIDTH, Visual.HEIGHT, false);
        g.setColor(Color.WHITE);
        for (Point p : points) {
            g.fill(new Ellipse2D.Double(
                    map(p.x(), 0, Math.PI*4, 0, WIDTH)-5,
                    p.y() * HEIGHT-5,
                    10,
                    10));
        }

        g.setColor(new Color(61, 61, 61));
        for(int i = 0; i < WIDTH; i += 1) {
            double x = map(i, 0, WIDTH, 0, Math.PI*4);
            double y = myFunction.apply(x) * HEIGHT;
            g.fill(new Ellipse2D.Double(i-5, y-5, 10,10));
        }

    }
    public static double map(double x, double inMin, double inMax, double outMin, double outMax){
        return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    }
        public void evaluateAll() {
        Arrays.stream(population.players()).forEach(Individual::evaluateAll);
//        Arrays.stream(population.players()).forEach(Individual::evaluateRange);
    }

    public static record Point(double x, double y, int label) {
    }
}
