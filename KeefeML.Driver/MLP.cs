namespace KeefeML.Driver
{
    public class MLP
    {
        private readonly Random _random = new();

        private double[][][] _weights;
        private double[][] _biases;
        private double[][] _outputs;
        private double[][] _deltas;

        private double[] _originalInput;

        public MLP(int[] structure)
        {
            // We need one of each per layer
            _biases = new double[structure.Length - 1][];
            _outputs = new double[structure.Length - 1][];
            _deltas = new double[structure.Length - 1][];
            _weights = new double[structure.Length - 1][][];

            _originalInput = [];

            for (var i = 0; i < structure.Length - 1; i++)
            {
                //  We need one of each of these per neuron
                _biases[i] = new double[structure[i + 1]];
                _outputs[i] = new double[structure[i + 1]];
                _deltas[i] = new double[structure[i + 1]];
                _weights[i] = new double[structure[i + 1]][];

                // setup each neuron in layer i (0, which is the first hidden layer)
                for(var n = 0; n < structure[i + 1]; n ++)
                {
                    //  He initialization of weights
                    var scale = Math.Sqrt(2.0 / structure[i]);

                    // Number of weights is equal to the number of neurons in the previous layer
                    // in this case, we are working at i + 1, so i is the previous layer
                    _weights[i][n] = new double[structure[i]];

                    //  Set the bias for this neuron
                    _biases[i][n] = _random.NextDouble() * scale;

                    for(var w = 0; w < structure[i]; w ++)
                    {
                        _weights[i][n][w] = _random.NextDouble() * scale;
                    }
                }
            }
        }

        public double[] FeedForward(double[] features)
        {
            _originalInput = features;

            // there is no actual input layer, so we start at the first hidden layer
            for(var l = 0; l < _weights.Length; l ++)
            {
                for(var n = 0; n < _weights[l].Length; n ++)
                {
                    //  We could also save some time here and set this equal to the bias
                    _outputs[l][n] = 0.0;

                    for(var w = 0; w < _weights[l][n].Length; w ++)
                    {
                        //  Since there is no input layer, we use the features for the first hidden layer
                        //  and then the outputs of the previous layers for all subsequent calculations
                        _outputs[l][n] += _weights[l][n][w]
                            * l == 0
                            ? features[w]
                            : _outputs[l - 1][w];                            
                    }

                    if (l < _weights.Length - 1)
                    {
                        // Perform the activation for all hidden layers - Tanh here because we're not going to normalize the data
                        _outputs[l][n] = Math.Tanh(_outputs[l][n] + _biases[l][n]);
                    }
                    else
                    {
                        // Use a sigmoid for the output layer
                        _outputs[l][n] = (1 / (1 + Math.Exp(-_outputs[l][n])));
                    }
                }
            }

            return _outputs[^1];
        }

        public double FindGradients(double[] expected)
        {
            var error = 0.0;
            for(var i = 0; i < expected.Length; i ++)
            {
                error += Math.Pow(expected[i] - _outputs[^1][i], 2) / expected.Length;
            }            

            // Notice that we're going backwards here
            for(var l = _weights.Length - 1; l >= 0; l --)
            {
                for(var n = 0; n < _weights[l].Length; n ++)
                {
                    if (l == _weights.Length - 1)
                    {
                        _deltas[^1][n] = error;
                    }
                    else
                    {
                        _deltas[l][n] = 0.0;
                        // Iterate trhough the connected neurons in the NEXT layer
                        for(var cn = 0; cn < _weights[l + 1].Length; cn ++)
                        {
                            // cn == connected neuron in the NEXT layer
                            // the result here is the connected weight X the delta of the neuron in which the weight resides
                            _deltas[l][n] += _weights[l + 1][cn][n] * _deltas[l + 1][cn];
                        }

                        // After we've summed the weights, we multiply them by the derivative of whatever activation we used
                        _deltas[l][n] += _deltas[l][n] * (1 - Math.Tanh(_outputs[l][n]) * Math.Tanh(_outputs[l][n]));
                    }
                }
            }

            //  We used the mean squared error to determine the loss.  We would track this
            //  over time if we were actually trying to make this thing learn something
            return error;
        }

        public void Adjust(double learningRate)
        {
            for(var l = 0; l < _weights.Length; l ++)
            {
                for(var n = 0; n < _weights[l].Length; n ++)
                {
                    for(var w = 0; w < _weights[l][n].Length; w ++)
                    {

                        var adjustment = learningRate * _deltas[l][n]
                            * l == 0
                            ? _originalInput[w]   // The initial features since this is the first hidden layer
                            : _outputs[l - 1][w]; // What the Neuron received

                        _weights[l][n][w] -= adjustment;
                    }
                }
            }
        }
    }
}
