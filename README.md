# DesertLandMLP
Simple MultiLayerPerceptron with multidimensionnal NDArray backend written in C#.

### Output

```
Hello World! Xor MLP.
Backend NDArray<Single>

Summary
Input Shape:2
Layer: Dense      Parameters:  24 Weights[In: 2 -> Out:8]
Layer: TANH       Parameters:   0 Weights[In: 8 -> Out:8]
Layer: Dense      Parameters:   9 Weights[In: 8 -> Out:1]
Layer: SIGMOID    Parameters:   0 Weights[In: 1 -> Out:1]
Output Shape:1
Total Parameters:33

Training Data. X Shape: 4x2; y Shape: 4x1
Start Training...
Epochs     0/1000 Loss:0.694496 Acc:0.5000
Epochs   100/1000 Loss:0.133947 Acc:1.0000
Epochs   200/1000 Loss:0.026717 Acc:1.0000
Epochs   300/1000 Loss:0.013081 Acc:1.0000
Epochs   400/1000 Loss:0.008362 Acc:1.0000
Epochs   500/1000 Loss:0.006050 Acc:1.0000
Epochs   600/1000 Loss:0.004699 Acc:1.0000
Epochs   700/1000 Loss:0.003821 Acc:1.0000
Epochs   800/1000 Loss:0.003208 Acc:1.0000
Epochs   900/1000 Loss:0.002758 Acc:1.0000
Epochs  1000/1000 Loss:0.002414 Acc:1.0000
End Training.
Time:185 ms

Prediction
[0 0] = [0] -> [0.000860]
[1 0] = [1] -> [0.997069]
[0 1] = [1] -> [0.997623]
[1 1] = [0] -> [0.003462]
```
