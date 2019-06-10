using System;
using System.Collections.Generic;
using System.Linq;

namespace DesertLandMLP
{
    public class Network<Type>
    {
        public readonly IOptimizer<Type> optimizer;
        public readonly ILoss<Type> lossFunction;
        public List<Layer<Type>> layers = new List<Layer<Type>>();

        public Network(IOptimizer<Type> optimizer, ILoss<Type> loss)
        {
            this.optimizer = optimizer;
            lossFunction = loss;
        }

        public void SetTrainable() => layers.ForEach(l => l.trainable = true);

        public void AddLayer(Layer<Type> layer)
        {
            if (layers.Count != 0)
                layer.SetInputShape(layers.Last().Outputs);

            layer.Initialize(optimizer);
            layers.Add(layer);
        }

        public NDArray<Type> ForwardPass(NDArray<Type> X, bool isTraining = true)
        {
            var layerOutput = new NDArray<Type>(X);
            foreach (var layer in layers)
                layerOutput = layer.Forward(layerOutput, isTraining);

            return layerOutput;
        }

        public void BackwardPass(NDArray<Type> lossGrad)
        {
            foreach (var layer in layers.Reverse<Layer<Type>>())
                lossGrad = layer.Backward(lossGrad);
        }

        public NDArray<Type> Predict(NDArray<Type> X) => ForwardPass(X, false);

        (double, double) TestOnBatch(NDArray<Type> X, NDArray<Type> y)
        {
            var yp = ForwardPass(X, false);
            var loss = NumDN.Mean(lossFunction.Loss(y, yp));
            var acc = lossFunction.Acc(y, yp);
            return (loss, acc);
        }

        public (double, double) TrainOnBatch(NDArray<Type> X, NDArray<Type> y)
        {
            var yp = ForwardPass(X);
            var loss = NumDN.Mean(lossFunction.Loss(y, yp));
            var acc = lossFunction.Acc(y, yp);
            var lossGrad = lossFunction.Grad(y, yp);
            BackwardPass(lossGrad);

            return (loss, acc);
        }

        public void Fit(NDArray<Type> X, NDArray<Type> y, int epochs, int batchSize = 64, int displayEpochs = 1)
        {
            Console.WriteLine("Start Training...");
            var batchData = BatchIterator(X, y, batchSize);

            for (int k = 0; k <= epochs; ++k)
            {
                List<double> losses = new List<double>();
                List<double> accs = new List<double>();

                foreach (var batch in batchData)
                {
                    var (loss, acc) = TrainOnBatch(batch.Item1, batch.Item2);
                    losses.Add(loss);
                    accs.Add(acc);
                }

                if (k % displayEpochs == 0)
                    Console.WriteLine("Epochs {0,5}/{1} Loss:{2:0.000000} Acc:{3:0.0000}", k, epochs, losses.Average(), accs.Average());
            }
            Console.WriteLine("End Training.");
        }

        public void Summary()
        {
            Console.WriteLine("Summary");
            Console.WriteLine($"Input Shape:{layers[0].Inputs}");
            int tot = 0;
            foreach (var layer in layers)
            {
                Console.WriteLine($"Layer: {layer.Name,-10} Parameters: {layer.Parameters,3} Nodes[In:{layer.Inputs,2} -> Out:{layer.Outputs}]");
                tot += layer.Parameters;
            }

            Console.WriteLine($"Output Shape:{layers.Last().Outputs}");
            Console.WriteLine($"Total Parameters:{tot}");
            Console.WriteLine();
        }

        public static List<(NDArray<Type>, NDArray<Type>)> BatchIterator(NDArray<Type> X, NDArray<Type> y, int batchSize)
        {
            int nbSamples = X.Shape[0];
            var shapeX = X.Shape.ToArray();
            var shapeY = y.Shape.ToArray();
            int nb = nbSamples / Math.Min(nbSamples, batchSize);
            var rg = Enumerable.Range(0, nb).Select(i => i * batchSize).ToList();
            List<(NDArray<Type>, NDArray<Type>)> data = new List<(NDArray<Type>, NDArray<Type>)>();

            foreach (var i in rg)
            {
                var (begin, end) = (i, Math.Min(i + batchSize, nbSamples));
                shapeX[0] = end - begin;
                shapeY[0] = end - begin;

                NDArray<Type> X0 = NDArray<Type>.Zeros(shapeX);
                NDArray<Type> y0 = NDArray<Type>.Zeros(shapeY);
                for (int j = begin, m = 0; j < end; ++j, ++m)
                {
                    X0[m] = X[j];
                    y0[m] = y[j];
                }

                data.Add((X0, y0));
            }

            return data;
        }
    }
}
