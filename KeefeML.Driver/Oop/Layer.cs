namespace KeefeML.Driver.Oop
{
    public class Layer
    {
        private readonly Random _random = new();

        public Neuron[] Neurons { get; set; }
        

        public Layer(int size, int connections)
        {
            Neurons = new Neuron[size];
            var scale = Math.Sqrt(2.0 / connections);
            for(var i= 0;  i < Neurons.Length; i++)
            {
                var weights = new List<double>();
                for(var j = 0; j < connections; j++)
                {
                    weights.Add(_random.NextDouble() * scale);
                }
                Neurons[i] = new Neuron
                {
                    Weights = [.. weights],
                    Bias = _random.NextDouble() * scale
                };
                weights.Clear();
            }
        }

        public double[] FeedForward(double[] features)
        {
            var outputs = new double[Neurons.Length];
            for(var i = 0;  i < Neurons.Length; i++)
            {
                outputs[i] = Sigmoid(Neurons[i].Sum(features));
            }
            return outputs;
        }


        public double Sigmoid(double value)
        {
            return 1 / (1 + (Math.Exp(-value)));
        }

        public double SigmoidPrime(double value)
        {
            return Sigmoid(value) * (1 - Sigmoid(value));
        }
    }
}
