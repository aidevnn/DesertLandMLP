# DesertLandMLP
Simple MultiLayerPerceptron with NumPy like backend written in C# for computing.

## Output

```
Hello World! Xor MLP
Summary
Input Shape:2
Layer: Dense      Parameters:  24 Nodes[In: 2 -> Out:8]
Layer: TANH       Parameters:   0 Nodes[In: 8 -> Out:8]
Layer: Dense      Parameters:   9 Nodes[In: 8 -> Out:1]
Layer: SIGMOID    Parameters:   0 Nodes[In: 1 -> Out:1]
Output Shape:1
Total Parameters:33

X Shape: 4x2
y Shape: 4x1
Start Training...
Epochs     0/1000 Loss:0.697378 Acc:0.2500
Epochs   100/1000 Loss:0.552347 Acc:1.0000
Epochs   200/1000 Loss:0.074435 Acc:1.0000
Epochs   300/1000 Loss:0.027877 Acc:1.0000
Epochs   400/1000 Loss:0.016336 Acc:1.0000
Epochs   500/1000 Loss:0.011361 Acc:1.0000
Epochs   600/1000 Loss:0.008638 Acc:1.0000
Epochs   700/1000 Loss:0.006936 Acc:1.0000
Epochs   800/1000 Loss:0.005777 Acc:1.0000
Epochs   900/1000 Loss:0.004939 Acc:1.0000
Epochs  1000/1000 Loss:0.004308 Acc:1.0000
End Training.
Time:182 ms

Prediction
[0 0] = [0] -> [0.004407]
[1 0] = [1] -> [0.994760]
[0 1] = [1] -> [0.996010]
[1 1] = [0] -> [0.003535]
```
