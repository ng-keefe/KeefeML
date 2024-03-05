namespace KeefeML.Driver.Oop
{
    public class Network
    {
        public Layer[] Layers { get; set; }
        public double[][] Outputs { get; set; }

        public Network(int[] structure)
        {
            Outputs = new double[structure.Length][];
            Layers = new Layer[structure.Length - 1];
            for(var i = 1; i < structure.Length; i ++)
            {
                Layers[i] = new Layer(i, i - 1);
            }
        }

        public double[] FeedForward(double[] features)
        {
            Outputs[0] = features;
            for(var i = 0; i < Layers.Length; i ++)
            {
                Outputs[i + 1] = Layers[i].FeedForward(Outputs[i]);
            }
            return Outputs[^1];
        }
    }
}
