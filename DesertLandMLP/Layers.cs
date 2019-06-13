using System;
using System.Collections.Generic;

namespace DesertLandMLP
{
    public abstract class Layer<Type>
    {
        public string Name;
        public bool trainable = true;
        public int Inputs { get; set; }
        public int Outputs { get; set; }
        public abstract void SetInputShape(int inputShape);
        public abstract int Parameters { get; }
        public abstract void Initialize(IOptimizer<Type> optimizer = null);
        public abstract NDArray<Type> Forward(NDArray<Type> X, bool isTraining);
        public abstract NDArray<Type> Backward(NDArray<Type> accumGrad);
    }

    public class ActivationLayer<Type> : Layer<Type>
    {
        static Dictionary<string, IActivation<Type>> Activations;

        static ActivationLayer()
        {
            Activations = new Dictionary<string, IActivation<Type>>();
            Activations["identity"] = new IdentityActivation<Type>() as IActivation<Type>;
            Activations["sigmoid"] = new SigmoidActivation<Type>() as IActivation<Type>;
            Activations["tanh"] = new TanhActivation<Type>() as IActivation<Type>;
            Activations["softmax"] = new SoftmaxActivation<Type>() as IActivation<Type>;
        }

        private readonly IActivation<Type> activation;
        private NDArray<Type> layerInput;

        public override int Parameters => 0;

        public ActivationLayer(string name)
        {
            activation = Activations[name];
            Name = name.ToUpper();
        }

        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            return accumGrad * activation.Gradient(layerInput);
        }

        public override void Initialize(IOptimizer<Type> optimizer = null) { }

        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining)
        {
            layerInput = new NDArray<Type>(X);
            return activation.Function(X);
        }

        public override void SetInputShape(int inputShape)
        {
            Inputs = Outputs = inputShape;
        }
    }

    public class IdentityLayer<Type> : ActivationLayer<Type>
    {
        public IdentityLayer() : base("identity") { }
    }

    public class SigmoidLayer<Type> : ActivationLayer<Type>
    {
        public SigmoidLayer() : base("sigmoid") { }
    }

    public class TanhLayer<Type> : ActivationLayer<Type>
    {
        public TanhLayer() : base("tanh") { }
    }

    public class SoftmaxLayer<Type> : ActivationLayer<Type>
    {
        public SoftmaxLayer() : base("softmax") { }
    }

    public class DenseLayer<Type> : Layer<Type>
    {
        private NDArray<Type> W, w0, layerInput;
        private IOptimizer<Type> WOpt, w0Opt;

        public override int Parameters => NumDN.ShapeCapacity(W.Shape) + NumDN.ShapeCapacity(w0.Shape);

        public DenseLayer(int nodes, int? inputShape = null)
        {
            Outputs = nodes;
            Name = "Dense";
            if (inputShape.HasValue) Inputs = inputShape.Value;
        }

        public override void Initialize(IOptimizer<Type> optimizer)
        {
            double lim = 1.0 / Math.Sqrt(Inputs);

            W = NumDN.Uniform<Type>(-lim, lim, Inputs, Outputs);
            w0 = NDArray<Type>.Zeros(1, Outputs);

            WOpt = optimizer.Clone();
            w0Opt = optimizer.Clone();
        }

        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining = true)
        {
            layerInput = new NDArray<Type>(X);
            return NDArray<Type>.Dot(X, W) + w0;
        }

        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            var Wtmp = new NDArray<Type>(W);

            if (trainable)
            {
                var gW = NDArray<Type>.Dot(layerInput.T, accumGrad);
                var gw0 = (new NDArray<Type>(accumGrad.Shape)) + NumDN.Sum(accumGrad);

                W = WOpt.Update(W, gW);
                w0 = w0Opt.Update(w0, gw0);
            }

            var accumGrad0 = NDArray<Type>.Dot(accumGrad, W.T);
            return accumGrad0;
        }

        public override void SetInputShape(int inputShape)
        {
            Inputs = inputShape;
        }
    }
}
