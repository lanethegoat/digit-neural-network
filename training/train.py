"""
Trains a small neural network to read handwritten digits (0-9) and
writes three files:

  data/model.json      the trained weights and biases
  data/timelapse.json  60 snapshots from during training, with the
                       training pictures it was looking at each time
  data/samples.json    8 example training pictures of each digit

Network shape: 784 inputs (28x28 pixels) -> 32 neurons -> 16 neurons -> 10 outputs

Run:  python3 train.py /path/to/mnist-folder
Needs only numpy.
"""
import base64
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
DEFAULT_MNIST_DIR = os.path.join(HERE, "mnist")

LAYER_SIZES = [784, 32, 16, 10]
EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 0.002
SEED = 7

NUM_SNAPSHOTS = 60
NUM_SHOWCASE = 24        # test digits the timelapse follows
NUM_FED_SHOWN = 16       # training pictures saved with each snapshot


# ---------- loading MNIST ----------

def read_images(folder, filename):
    with gzip.open(os.path.join(folder, filename)) as f:
        pixels = np.frombuffer(f.read(), np.uint8, offset=16)
    return pixels.reshape(-1, 784).astype(np.float32) / 255.0


def read_labels(folder, filename):
    with gzip.open(os.path.join(folder, filename)) as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


def load_mnist(folder):
    train_images = read_images(folder, "train-images-idx3-ubyte.gz")
    train_labels = read_labels(folder, "train-labels-idx1-ubyte.gz")
    test_images = read_images(folder, "t10k-images-idx3-ubyte.gz")
    test_labels = read_labels(folder, "t10k-labels-idx1-ubyte.gz")
    return train_images, train_labels, test_images, test_labels


# ---------- augmentation ----------
# Drawings on an iPad are messier than MNIST, so after the first pass each
# training picture gets a small random rotation, zoom and shift.

# pixel coordinates measured from the middle of the 28x28 picture
grid_y, grid_x = np.mgrid[0:28, 0:28].astype(np.float32)
grid_y -= 13.5
grid_x -= 13.5


def augment(images, rng):
    n = images.shape[0]
    angle = rng.uniform(-0.25, 0.25, n)    # radians, about 14 degrees either way
    zoom = rng.uniform(0.85, 1.12, n)
    shift_x = rng.uniform(-2.5, 2.5, n)
    shift_y = rng.uniform(-2.5, 2.5, n)

    # for each output pixel, work out where to sample from in the original
    cos = (np.cos(angle) / zoom)[:, None, None]
    sin = (np.sin(angle) / zoom)[:, None, None]
    src_x = cos * grid_x + sin * grid_y + 13.5 - shift_x[:, None, None]
    src_y = -sin * grid_x + cos * grid_y + 13.5 - shift_y[:, None, None]

    left = np.floor(src_x).astype(int)
    top = np.floor(src_y).astype(int)
    frac_x = src_x - left
    frac_y = src_y - top

    # one pixel of black border so samples just off the edge read as 0
    padded = np.zeros((n, 30, 30), np.float32)
    padded[:, 1:29, 1:29] = images.reshape(n, 28, 28)
    which = np.arange(n)[:, None, None]

    def pixel(y, x):
        return padded[which, np.clip(y + 1, 0, 29), np.clip(x + 1, 0, 29)]

    # bilinear interpolation between the 4 nearest pixels
    result = (pixel(top, left) * (1 - frac_x) * (1 - frac_y)
              + pixel(top, left + 1) * frac_x * (1 - frac_y)
              + pixel(top + 1, left) * (1 - frac_x) * frac_y
              + pixel(top + 1, left + 1) * frac_x * frac_y)
    return result.reshape(n, 784)


# ---------- the network ----------
# params is a list with one [weights, biases] pair per layer

def init_params(rng):
    params = []
    for n_in, n_out in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]):
        weights = rng.normal(0, np.sqrt(1 / n_in), (n_in, n_out)).astype(np.float32)
        biases = np.zeros(n_out, np.float32)
        params.append([weights, biases])
    return params


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)    # keeps exp() from overflowing
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, x):
    """Returns the activations of every layer. The last one is the 10 probabilities."""
    layers = [x]
    last = len(params) - 1
    for i, (weights, biases) in enumerate(params):
        z = layers[-1] @ weights + biases
        layers.append(softmax(z) if i == last else sigmoid(z))
    return layers


def backward(params, layers, labels):
    """Backpropagation for cross-entropy loss. Returns gradients shaped like params."""
    n = len(labels)
    error = layers[-1].copy()
    error[np.arange(n), labels] -= 1
    error /= n
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        weights = params[i][0]
        grads[i] = [layers[i].T @ error, error.sum(axis=0)]
        if i > 0:
            error = (error @ weights.T) * layers[i] * (1 - layers[i])
    return grads


def predict(params, x):
    return forward(params, x)[-1].argmax(1)


def accuracy(params, x, labels):
    return float((predict(params, x) == labels).mean())


# ---------- training ----------

class Adam:
    def __init__(self, params):
        self.m = [[np.zeros_like(p) for p in layer] for layer in params]
        self.v = [[np.zeros_like(p) for p in layer] for layer in params]
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        for layer in range(len(params)):
            for k in range(2):
                g = grads[layer][k]
                self.m[layer][k] = 0.9 * self.m[layer][k] + 0.1 * g
                self.v[layer][k] = 0.999 * self.v[layer][k] + 0.001 * g * g
                m_hat = self.m[layer][k] / (1 - 0.9 ** self.t)
                v_hat = self.v[layer][k] / (1 - 0.999 ** self.t)
                params[layer][k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def learning_rate(epoch):
    # halve it for the last 2 passes so it settles down
    if epoch >= EPOCHS - 2:
        return LEARNING_RATE * 0.5
    return LEARNING_RATE * 1.0


def batches(rng, train_images, train_labels, use_augmentation=True):
    """Yields (epoch, images, labels) for every batch. Augmentation starts on pass 2."""
    steps_per_epoch = len(train_images) // BATCH_SIZE
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_images))
        for k in range(steps_per_epoch):
            picks = order[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
            images = train_images[picks]
            if use_augmentation and epoch > 0:
                images = augment(images, rng)
            yield epoch, images, train_labels[picks]


def log_spaced_steps(total, count):
    """Step numbers bunched up near the start, where the network changes fastest."""
    steps = np.unique(np.round(np.geomspace(1, total, count))).astype(int)
    return set(steps.tolist())


# ---------- saving ----------

def to_base64(image):
    """Pixels 0..1 -> bytes 0..255 -> base64 text, so pictures fit in JSON."""
    pixels = np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8)
    return base64.b64encode(pixels.tobytes()).decode()


def save_json(name, obj):
    with open(os.path.join(DATA_DIR, name + ".json"), "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def pick_showcase(test_labels, rng):
    """3 test pictures of each digit, shuffled, keeping the first 24."""
    picks = []
    for digit in range(10):
        picks += list(np.where(test_labels == digit)[0][:3])
    return rng.permutation(picks)[:NUM_SHOWCASE]


def main():
    mnist_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MNIST_DIR
    train_images, train_labels, test_images, test_labels = load_mnist(mnist_dir)
    rng = np.random.default_rng(SEED)

    showcase = pick_showcase(test_labels, rng)
    showcase_images = test_images[showcase]
    showcase_labels = test_labels[showcase]

    params = init_params(rng)
    optimizer = Adam(params)
    steps_per_epoch = len(train_images) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    snapshot_steps = log_spaced_steps(total_steps, NUM_SNAPSHOTS - 1)
    snapshots = []

    def take_snapshot(step, loss, images, labels):
        # what the network guesses for the training pictures it was just fed
        probs = forward(params, images[:NUM_FED_SHOWN])[-1]
        fed = []
        for i in range(min(NUM_FED_SHOWN, len(images))):
            fed.append({"img": to_base64(images[i]), "label": int(labels[i]),
                        "guess": int(probs[i].argmax()), "conf": round(float(probs[i].max()), 2)})

        # first-layer weights as greyscale pictures (grey = 0, white = positive)
        w1 = params[0][0]
        scale = max(abs(np.percentile(w1, 1)), abs(np.percentile(w1, 99)))
        w1_pictures = np.clip(w1.T / scale * 0.5 + 0.5, 0, 1)

        snap = {
            "step": step,
            "seen": step * BATCH_SIZE,
            "epoch": round(step / steps_per_epoch, 2),
            "loss": round(float(loss), 4),
            "accuracy": round(accuracy(params, test_images, test_labels), 4),
            "probs": np.round(forward(params, showcase_images)[-1], 3).tolist(),
            "fed": fed,
            "w1": to_base64(w1_pictures),
        }
        snapshots.append(snap)
        print(f"step {step:6d}  epoch {snap['epoch']:5.2f}  loss {snap['loss']:.3f}  "
              f"test acc {snap['accuracy'] * 100:.1f}%", flush=True)

    # before any training, guessing is basically random, so loss = log(10)
    take_snapshot(0, np.log(10), train_images[:NUM_FED_SHOWN], train_labels[:NUM_FED_SHOWN])
    smooth_loss = None
    for epoch, images, labels in batches(rng, train_images, train_labels):
        layers = forward(params, images)
        loss = -np.log(layers[-1][np.arange(len(labels)), labels] + 1e-9).mean()
        smooth_loss = loss if smooth_loss is None else 0.98 * smooth_loss + 0.02 * loss
        grads = backward(params, layers, labels)
        optimizer.step(params, grads, learning_rate(epoch))
        if optimizer.t in snapshot_steps:
            take_snapshot(optimizer.t, smooth_loss, images, labels)

    guesses = predict(params, test_images)
    final_accuracy = accuracy(params, test_images, test_labels)
    per_digit = [round(float((guesses[test_labels == d] == d).mean()), 4) for d in range(10)]
    print(f"final test accuracy {final_accuracy * 100:.2f}%  per digit {per_digit}")

    os.makedirs(DATA_DIR, exist_ok=True)
    save_json("model", {
        "layers": LAYER_SIZES,
        "activation": "sigmoid",
        "output": "softmax",
        "accuracy": round(final_accuracy, 4),
        "perDigit": per_digit,
        "trainImages": len(train_images),
        "testImages": len(test_images),
        "epochs": EPOCHS,
        "weights": [np.round(w, 4).tolist() for w, _ in params],
        "biases": [np.round(b, 4).tolist() for _, b in params],
    })
    save_json("timelapse", {
        "layers": LAYER_SIZES,
        "batch": BATCH_SIZE,
        "totalSteps": total_steps,
        "showcase": {"images": [to_base64(img) for img in showcase_images],
                     "labels": showcase_labels.tolist()},
        "snapshots": snapshots,
    })
    save_json("samples", {
        str(d): [to_base64(train_images[i]) for i in np.where(train_labels == d)[0][:8]]
        for d in range(10)
    })


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"done in {time.time() - start:.0f}s")
