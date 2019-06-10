using System;
using System.Diagnostics;

namespace DesertLandMLP
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World! Xor MLP");

            var net = new Network<double>(new SGD<double>(0.15), new CrossEntropy<double>());
            net.AddLayer(new DenseLayer<double>(8, inputShape: 2));
            //net.AddLayer(new TanhLayer());
            //net.AddLayer(new DenseLayer(6));
            net.AddLayer(new TanhLayer<double>());
            net.AddLayer(new DenseLayer<double>(1));
            net.AddLayer(new SigmoidLayer<double>());
            net.Summary();

            var X = new NDArray<double>(new double[4, 2] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } });
            var y = new NDArray<double>(new double[4, 1] { { 0 }, { 1 }, { 1 }, { 0 } });

            Console.WriteLine("X Shape: {0}", X.Shape.Glue("x"));
            Console.WriteLine("y Shape: {0}", y.Shape.Glue("x"));

            var sw = Stopwatch.StartNew();
            net.Fit(X, y, 1000, displayEpochs: 100);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");
            Console.WriteLine();

            Console.WriteLine("Prediction");
            var pred = net.Predict(X).Apply(x => Math.Round(x, 6));
            for (int k = 0; k < X.Shape[0]; ++k)
                Console.WriteLine($"{X[k]} = {y[k]} -> {pred[k]}");
        }
    }
}
