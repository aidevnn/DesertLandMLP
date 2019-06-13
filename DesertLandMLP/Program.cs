using System;
using System.Diagnostics;

namespace DesertLandMLP
{
    class MainClass
    {
        static void Run<Type>()
        {
            Console.WriteLine($"Hello World! Xor MLP.");
            Console.WriteLine($"Backend NDArray<{typeof(Type).Name}>");
            Console.WriteLine();

            var net = new Network<Type>(new SGD<Type>(0.015), new CrossEntropy<Type>());
            net.AddLayer(new DenseLayer<Type>(8, inputShape: 2));
            //net.AddLayer(new TanhLayer());
            //net.AddLayer(new DenseLayer(6));
            net.AddLayer(new TanhLayer<Type>());
            net.AddLayer(new DenseLayer<Type>(1));
            net.AddLayer(new SigmoidLayer<Type>());
            net.Summary();

            var X = (new NDArray<double>(new double[4, 2] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } })).Cast<Type>();
            var y = (new NDArray<double>(new double[4, 1] { { 0 }, { 1 }, { 1 }, { 0 } })).Cast<Type>();

            Console.WriteLine("Training Data. X Shape: {0}; y Shape: {1}", X.Shape.Glue("x"), y.Shape.Glue("x"));

            var sw = Stopwatch.StartNew();
            net.Fit(X, y, 10000, displayEpochs: 1000);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");
            Console.WriteLine();

            Console.WriteLine("Prediction");
            var pred = net.Predict(X).Apply(x => Math.Round(Convert.ToDouble(x), 6));
            for (int k = 0; k < X.Shape[0]; ++k)
                Console.WriteLine($"{X[k]} = {y[k]} -> {pred[k]}");
        }

        public static void Main(string[] args)
        {
            //Run<float>();
            Run<double>();
        }
    }
}
