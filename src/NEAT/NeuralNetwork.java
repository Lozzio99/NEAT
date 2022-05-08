package NEAT;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.*;
import static NEAT.Example.Settings.RANDOM;

public class NeuralNetwork implements Cloneable {

    private final int id;
    private final int numInputs;
    private final int numOutputs;
    private final List<Node> nodes;
    private final List<Connection> connections;
    private final boolean offSpring;
    private int layers;
    private int nextNode;
    private NeuralNetwork(int id, int numInputs, int numOutputs) {
        this(id,numInputs,numOutputs,false);
    }
    private NeuralNetwork(int id, int numInputs, int numOutputs, boolean offSpring) {
        this.id = id;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.offSpring = offSpring;
        this.layers = 2;
        this.nextNode = 0;
        this.nodes = new ArrayList<>();
        this.connections = new ArrayList<>();

        if (!offSpring) this.generateNodes();
    }
    public static NeuralNetwork createNetwork(int numInputs, int numOutputs) {
        NeuralNetwork n = new NeuralNetwork(0,numInputs,numOutputs);
        n.generateNetwork();
        return n;
    }

    public double[] feedforward(double[] input) {
        this.generateNetwork();
        this.nodes.forEach(n -> n.inputSum = n.outputValue = 0);
        for (int i = 0; i< this.numInputs; i++) this.nodes.get(i).outputValue = input[i];
        double[] result = new double[this.numOutputs];
        for (int i = 0,k = 0; i< this.nodes.size() && k < this.numOutputs;i++ ){
            Node n = nodes.get(i);
            n.engage();
            if (n.output()) {
                result[k++] = n.outputValue();
            }
        }
        return result;
    }
    private void generateNodes() {
        for (int i = 0; i< this.numInputs; i++){
            this.nodes.add(new Node(this.nextNode++,0,false));
        }

        for (int i = 0; i< this.numOutputs;i++) {
            this.nodes.add(new Node(this.nextNode++,1,true));
        }

        for (int i = 0; i< this.numInputs; i++){
            for (int j = this.numInputs; j< this.numOutputs+this.numInputs; j++) {
                double weight = RANDOM.nextDouble() * this.numInputs * sqrt(2. / this.numInputs);
                Connection c =  new Connection(this.nodes.get(i), this.nodes.get(j), weight);
                c.enable();
                this.connections.add(c);
            }
        }
    }
    public void generateNetwork() {
        this.nodes.forEach(n -> n.outputConnections.clear());
        this.connections.forEach(c -> c.fromNode().outputConnections.add(c));
        this.sortByLayer();
    }
    private void sortByLayer() {
        this.nodes.sort(Comparator.comparingInt(n -> n.layer));
    }
    public void mutate() {
        String mutationType = "";
        if (RANDOM.nextDouble() < 0.6) {
            mutationType += " Weight ";
            this.connections.forEach(Connection::mutateWeight);
        }

        if (RANDOM.nextDouble() < 0.6) {
            mutationType += " Bias ";
            this.nodes.forEach(Node::mutateBias);
        }

        if (RANDOM.nextDouble() < 0.4) {
            mutationType += " Activation ";
            this.nodes.forEach(Node::mutateActivation);
        }


        if (RANDOM.nextDouble() < 0.05) {
            mutationType += " AddConnection ";
            this.addConnection();
        }

        if (RANDOM.nextDouble() < 0.05) {
            mutationType += " AddNode ";
            this.addNode();
        }

        if (RANDOM.nextDouble() < 0.1) {
            mutationType += " RemoveNode ";
            this.removeNode();
        }

        if (RANDOM.nextDouble() < 0.2) {
            mutationType += " Enabled ";
            this.mutateEnable();
        }

        if (RANDOM.nextDouble() < 0.4) {
            mutationType += " Disabled ";
            this.mutateDisable();
        }


//        System.out.println(mutationType);

    }
    private void removeNode() {
        int nodeIndex = RANDOM.nextInt(this.nodes.size());
        Node picked = this.nodes().get(nodeIndex);
        if (picked.output() || picked.layer() == 0) return;
        this.nodes().remove(picked);

        boolean removeLayer = this.nodes().stream().noneMatch(n -> n.layer == picked.layer());

        List<Node> incoming = this.connections().stream().filter(c -> c.toNode().equals(picked)).map(Connection::fromNode).collect(Collectors.toList());
        List<Node> outgoing = this.connections().stream().filter(c -> c.fromNode().equals(picked)).map(Connection::toNode).collect(Collectors.toList());

        if (incoming.isEmpty() || outgoing.isEmpty()) {
            System.err.println("Disconnected node");
        }

        this.connections().removeIf(c -> c.toNode().equals(picked) || c.fromNode().equals(picked));
        for (Node n : incoming) {
            if (connections().stream().noneMatch(c -> c.fromNode().equals(n))) {
                Node out = outgoing.get(RANDOM.nextInt(outgoing.size()));
                Connection replacement = new Connection(n,out, RANDOM.nextDouble(),true);
                this.connections().add(replacement);
            }
        }
        for (Node n : outgoing) {
            if (connections().stream().noneMatch(c -> c.toNode().equals(n))) {
                Node in = incoming.get(RANDOM.nextInt(incoming.size()));
                Connection replacement = new Connection(in,n, RANDOM.nextDouble(),true);
                this.connections().add(replacement);
            }
        }

        if (removeLayer) {
            this.nodes().stream().filter(n -> n.layer() > picked.layer()).forEach(n -> n.layer--);
            this.layers--;
        }
        this.generateNetwork();
    }
    private void addNode() {
        int connectionIndex = RANDOM.nextInt(this.connections.size());
        Connection picked = this.connections.get(connectionIndex);
        this.connections.remove(picked);
        Node newNode = new Node(this.nextNode, picked.fromNode().layer()+1,false);

        boolean addingLayer = picked.toNode().layer() - picked.fromNode().layer() == 1;
        if (addingLayer) {
            int maxLayer = -1;
            int newLayer = picked.toNode().layer();
            for (Node n : this.nodes) {
                if (n.layer >= newLayer) n.layer++;
                if (n.layer > maxLayer) maxLayer = n.layer;
            }
            this.layers = maxLayer+1;
        }


        Connection c1 = new Connection(picked.fromNode(),newNode,1,true);
        Connection c2 = new Connection(newNode,picked.toNode(), picked.weight(),true);

        this.connections.add(c1);
        this.connections.add(c2);
        this.nodes.add(newNode);
        this.nextNode++;
    }
    private void addConnection() {
        if (this.fullyConnected()) return;
        int n1 = RANDOM.nextInt(this.nodes.size());
        int n2 = RANDOM.nextInt(this.nodes.size());

        while ((this.nodes.get(n1).layer == this.nodes.get(n2).layer) ||
                this.nodesConnected(this.nodes.get(n1), this.nodes.get(n2))) {
            n1 = RANDOM.nextInt(this.nodes.size());
            n2 = RANDOM.nextInt(this.nodes.size());
        }

        if (this.nodes.get(n1).layer > this.nodes.get(n2).layer){
            var tmp = n1;
            n1 = n2;
            n2 = tmp;
        }
        Connection newConnection = new Connection(
                this.nodes.get(n1),
                this.nodes.get(n2),
                RANDOM.nextDouble() * this.numInputs * sqrt(2./ this.numInputs));

        newConnection.enable();
        this.connections.add(newConnection);
        this.generateNetwork();
    }
    private void mutateEnable() {
        this.connections.get(RANDOM.nextInt(connections.size())).enable();
    }
    private void mutateDisable() {
        Connection c = this.connections.get(RANDOM.nextInt(connections.size()));
        if (!c.toNode().output()) c.disable();
    }
    private int getIndex(int node) {
        return IntStream.range(0, this.nodes.size()).
                filter(i -> this.nodes.get(i).index() == node).
                findFirst().orElse(-1);
    }
    private int commonConnection(int innovation, List<Connection> connections) {
        return IntStream.range(0,connections.size()).
                filter(i -> connections.get(i).innovationNumber() == innovation).
                findFirst().orElse(-1);
    }
    private int calculateWeight() {
        return this.connections.size() + this.nodes.size();
    }
    public boolean nodesConnected(Node n1, Node n2) {
        return this.connections.stream().anyMatch(c ->
                (c.fromNode().equals(n1) && c.toNode().equals(n2)) ||
                        (c.fromNode().equals(n2) && c.toNode().equals(n1)));
    }
    public boolean fullyConnected() {
        int maxConnections = 0;
        double[] nodesPerLayer = new double[this.nodes.size()];
        Arrays.fill(nodesPerLayer,-1);

        this.nodes.forEach(n ->  {
            if (nodesPerLayer[n.layer] < 0) nodesPerLayer[n.layer] = 1;
            else nodesPerLayer[n.layer] ++;
        });

        for (int i = 0; i< this.layers-1; i++) {
            for (int j = i+1; j< this.layers; j++){
                maxConnections+= nodesPerLayer[i]*nodesPerLayer[j];
            }
        }

        return maxConnections == this.connections.size();
    }
    public NeuralNetwork crossover(NeuralNetwork partner) {
        NeuralNetwork offSpring = new NeuralNetwork(
                max(this.id,partner.id)+1,
                this.numInputs,
                this.numOutputs,
                true
        );
        offSpring.nextNode = this.nextNode;

        for (int i = 0; i< this.nodes.size(); i++) {
            Node n = this.nodes.get(i).clone();
            assert n != null;
            if (n.output()) {
                Node partnerNode = partner.nodes.get(partner.getIndex(n.index()));
                if (RANDOM.nextDouble() > 0.5){
                    n.activationFunction = partnerNode.activationFunction;
                    n.bias = partnerNode.bias;
                }
            }
            offSpring.nodes.add(n);
        }

        for(int i = 0; i < this.connections.size(); i++) {
            int index = this.commonConnection(this.connections.get(i).innovationNumber(), partner.connections);

            Connection oldC;
            if(index != -1) oldC = Math.random() > 0.5 ? this.connections.get(i) : partner.connections.get(index);
            else oldC = this.connections.get(i);

            Node fromNode = offSpring.nodes.get(offSpring.getIndex(oldC.fromNode().index()));
            Node toNode = offSpring.nodes.get(offSpring.getIndex(oldC.toNode().index()));

            if(fromNode!= null && toNode != null)
                offSpring.connections.add(new Connection(fromNode, toNode,oldC.weight(),true));
        }

        offSpring.layers = this.layers;
        return offSpring;
    }

    @Override public String toString() {
        return """
                NeuralNetwork {
                    id=%d,
                    numInputs=%d,
                    numOutputs=%d,
                    nodes=%d,
                    layers=%d,
                    offSpring=%s
                    connections=%s,
                }""".formatted(id, numInputs, numOutputs, nodes.size(), layers, offSpring, connections);
    }
    @Override public NeuralNetwork clone() {
        try {
            NeuralNetwork clone = (NeuralNetwork) super.clone();
            clone.generateNetwork();
            return clone;
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }

    public int layers() {
        return this.layers;
    }
    public List<Node> nodes() {
        return this.nodes;
    }
    public List<Connection> connections() {
        return this.connections;
    }

    public static final class Node implements Cloneable {
        private final int index;
        private int layer;
        private final boolean output;
        private final List<Connection> outputConnections;
        private double inputSum;
        private double outputValue;
        private int activationFunction;
        private double bias;
        private Node(int index, int layer, boolean output) {
            this.index = index;
            this.layer = layer;
            this.output = output;
            this.inputSum = 0;
            this.outputValue = 0;
            this.activationFunction = RANDOM.nextInt(7);
            this.bias = RANDOM.nextDouble(-1,1);
            this.outputConnections = new ArrayList<>();
        }

        public int index() {
            return index;
        }
        public int layer() {
            return layer;
        }
        public boolean output() {
            return output;
        }
        public double outputValue() {
            return outputValue;
        }
        public double inputSum() {
            return inputSum;
        }
        public void mutateBias() {
            double r = random();
            if (r < 0.05)
                this.bias = RANDOM.nextDouble(-1,1);
            else
                this.bias += RANDOM.nextGaussian() / 2;
        }
        public void mutateActivation() {
            this.activationFunction = RANDOM.nextInt(7);
        }
        public boolean isConnectedTo(Node node) {
            if (node.layer() == this.layer) return false;
            if (node.layer < this.layer)
                return node.outputConnections.stream().anyMatch(c -> c.toNode().equals(this));
            else
                return this.outputConnections.stream().anyMatch(c -> c.toNode().equals(node));
        }
        public List<Connection> outputConnections() {
            return outputConnections;
        }

        @Override public boolean equals(Object obj) {
            if (obj == this) return true;
            if (obj == null || obj.getClass() != this.getClass()) return false;
            var that = (Node) obj;
            return this.index == that.index &&
                    this.layer == that.layer &&
                    this.output == that.output;
        }
        @Override public int hashCode() {
            return Objects.hash(index, layer, output);
        }
        @Override public String toString() {
            var c = outputConnections.stream().map(conn -> conn.toNode().index()).collect(Collectors.toList());
            return ("""
                    Node {
                                index=%d,
                                layer=%d,
                                output=%s,
                                outputConnections=%s,
                                inputSum=%s,
                                outputValue=%s,
                                activationFunction=%d,
                                bias=%s
                            }""").formatted(index, layer, output, c, inputSum, outputValue, activationFunction, bias);
        }
        @Override public Node clone() {
            try {
                Node n = (Node) super.clone();
                n.inputSum = this.inputSum;
                n.outputValue = this.outputValue;
                n.activationFunction = this.activationFunction;
                n.bias = this.bias;
                n.outputConnections.clear();
                n.outputConnections.addAll(this.outputConnections);
                return n;
            } catch (CloneNotSupportedException e) {
                e.printStackTrace();
            }
            return null;
        }
        public void engage() {
            if (this.layer != 0)
                this.outputValue = this.activation(this.inputSum + this.bias);

            for (Connection c : outputConnections){
                if (c.enabled()) c.toNode().inputSum += c.weight() * this.outputValue;
            }
        }
        public double activation(double x) {
            switch (activationFunction) {
                case 1 -> { return x; } //Identity
                case 2 -> { return x > 0 ? 1 : 0; } //Step
                case 3 -> { return tanh(x); } //Tanh
                case 4 -> { return x < 0 ? 0 : x; } //ReLu
                case 5 -> { return sin(x);} //Sin
                case 6 -> { return cos(x);} //Sin
                default -> { //Sigmoid
                    return 1 / (1 + Math.pow(Math.E, -4.9 * x));
                }
            }
        }
        public int activationKey() {
            return this.activationFunction;
        }
    }
    public static final class Connection implements Cloneable {
        private final Node fromNode;
        private final Node toNode;
        private double weight;
        private boolean enabled;

        private Connection(Node fromNode, Node toNode, double weight) {
            this(fromNode,toNode,weight,false);
        }

        private Connection(Node fromNode, Node toNode, double weight, boolean enabled) {
            this.fromNode = fromNode;
            this.toNode = toNode;
            this.weight = weight;
            this.enabled = enabled;
        }

        public void mutateWeight() {
            double r = random();
            if (r < 0.05)
                this.weight = RANDOM.nextDouble(-1,1);
            else
                this.weight += RANDOM.nextGaussian() / 2;
        }
        public int innovationNumber() {
            int i = this.fromNode.index();
            int j = this.toNode.index();
            return ((i+j-2) * (i+j-1) / 2 ) + i;
        }
        public void enable() {
            this.enabled = true;
        }
        public void disable() {this.enabled = false;}
        public boolean enabled() { return this.enabled;}
        public Node fromNode() {
            return fromNode;
        }
        public Node toNode() {
            return toNode;
        }
        public double weight() {
            return weight;
        }


        @Override public boolean equals(Object obj) {
            if (obj == this) return true;
            if (obj == null || obj.getClass() != this.getClass()) return false;
            var that = (Connection) obj;
            return Objects.equals(this.fromNode, that.fromNode) &&
                    Objects.equals(this.toNode, that.toNode) &&
                    Double.doubleToLongBits(this.weight) == Double.doubleToLongBits(that.weight);
        }
        @Override public int hashCode() {
            return Objects.hash(fromNode, toNode, weight);
        }

        @Override
        public String toString() {
            return """
                   Connection {
                            fromNode = %s,
                            toNode = %s,
                            weight = %s,
                            enabled = %s
                        }""".formatted(fromNode, toNode, weight, enabled);
        }

        @Override
        protected Connection clone()  {
            try {
                Connection clone = (Connection) super.clone();
                clone.enabled = this.enabled;
                return clone;
            } catch (CloneNotSupportedException e) {
                e.printStackTrace();
            }
            return null;
        }
    }

}
