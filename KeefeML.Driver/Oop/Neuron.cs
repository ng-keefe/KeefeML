namespace KeefeML.Driver.Oop
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }
        public double Delta { get; set; }

        public double Sum(double[] features)
        {
            var sum = 0.0;

            for(var i = 0; i < features.Length; i ++)
            {
                sum += features[i] * Weights[i];
            }

            return sum + Bias;
        }
    }
}
