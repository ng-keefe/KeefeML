namespace KeefeML.Driver
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var net = new MLP([2, 2, 1]);


            //  If we were training the network, we would have a data set 
            //  and pass each sample through adjusting the network after
            //  each iteration.  This network wasn't built to handle batches, 
            var outputs = net.FeedForward([1.2, -2.3]);
            var loss = net.FindGradients([1]);
            net.Adjust(0.1);
            Console.WriteLine(string.Join(",", outputs));
            Console.WriteLine($"The loss: {loss}");
        }
    }
}