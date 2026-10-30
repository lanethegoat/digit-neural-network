# Handwritten digit neural network

A small neural network that reads handwritten digits (0 to 9). I built it for a science fair project. This repo has the trained model, the code I used to train it, and my results, so you can train it yourself and check my numbers.

## The network

It takes a 28×28 picture (784 inputs), passes it through a layer of 32 neurons and then a layer of 16, and ends with 10 outputs, one per digit. The hidden layers use sigmoid and the output uses softmax. I trained it with Adam (learning rate 0.002, batch size 64) for 8 passes through the training set. During training, each picture gets a small random rotation, zoom and shift (augmentation).

It gets 95.0% of the 10,000 MNIST test digits right. It never saw those digits during training.

## The data

All training and testing used [MNIST](https://yann.lecun.com/exdb/mnist/), a public set of 70,000 handwritten digits from Yann LeCun, Corinna Cortes and Christopher Burges. 60,000 are for training and 10,000 for testing. I didn't use any other data.

MNIST isn't copied into this repo. To download it:

```
python3 mnist.py
```

## Files

|Path|What it is|
|-|-|
|`data/model.json`|The trained network: every weight and bias, plus test accuracy for each digit|
|`data/experiments.json`|Results from the experiments below (5 seeds, with and without augmentation, learning curves, confusion matrices)|
|`data/timelapse.json`|60 snapshots from during training, with the pictures it was training on at each one|
|`data/samples.json`|8 example training pictures of each digit|
|`graphs/`|Learning curve, accuracy per digit, and confusion matrix graphs, plus `summary.txt` with the statistics|
|`training/train.py`|Trains the network and writes `model.json`, `timelapse.json` and `samples.json`|
|`training/experiments.py`|Trains 10 networks (5 seeds, each with and without augmentation) and writes `experiments.json`|
|`training/graphs.py`|Draws the graphs from `experiments.json`|
|`mnist.py`|Downloads MNIST into `training/mnist/`|

## Training it yourself

You need Python 3 and numpy. The graphs also need matplotlib.

```
pip install -r requirements.txt
python3 _mnist.py
cd training
python3 train.py          # about 40 seconds, rewrites data/model.json
python3 experiments.py    # about 7 minutes, rewrites data/experiments.json
python3 graphs.py         # redraws graphs/
```

The random seed is fixed, so `train.py` should give you the same 95.0%.

## Results

||Average test accuracy (5 seeds)|
|-|-|
|With augmentation|95.13% (SD 0.23)|
|Without augmentation|96.15% (SD 0.11)|

I expected augmentation to help, but it lowered MNIST test accuracy by about 1 point (paired t-test, 95% CI −1.29 to −0.75). The network's most common mistake is reading a 4 as a 9. The full numbers are in `graphs/summary.txt`.

## Using the model

`data/model.json` stores `layers`, `weights` and `biases` as plain lists, so any language can load it. To guess a digit, scale the 28×28 picture so each pixel is between 0 and 1 (white digit on black) and flatten it into 784 numbers. For each layer, multiply by the weights and add the biases. Use sigmoid on the hidden layers and softmax on the last one. The largest of the 10 outputs is the guess.

## Credits

MNIST: Y. LeCun, C. Cortes and C. J. C. Burges.

