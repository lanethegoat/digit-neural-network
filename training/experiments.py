"""
Extra experiments for the paper, all on MNIST only:

  1. Control: the same network trained WITHOUT augmentation.
  2. Repeats: both versions trained with 5 different random seeds.
  3. Confusion matrix: which digit each test mistake was mistaken for.

Saves data/experiments.json. It reuses the network code from train.py, so
the math is exactly the same as the main run. Seed 7 with augmentation is
the main run, so it should give the same 95.0%.

Run:  python3 experiments.py /path/to/mnist-folder
Needs only numpy. Takes about 10 runs x 40 seconds.
"""
import json
import os
import sys
import time

import numpy as np

import train

SEEDS = [7, 1, 2, 3, 4]
NUM_CHECKS = 40    # test-accuracy checks per run, bunched up near the start


def run(seed, use_augmentation, train_images, train_labels, test_images, test_labels):
    rng = np.random.default_rng(seed)
    # train.py shuffles its 30 showcase digits before anything else. Do the same
    # shuffle here so the random numbers line up and seed 7 matches the main run.
    rng.permutation(30)

    params = train.init_params(rng)
    optimizer = train.Adam(params)
    total_steps = train.EPOCHS * (len(train_images) // train.BATCH_SIZE)
    check_steps = train.log_spaced_steps(total_steps, NUM_CHECKS)

    curve = [[0, train.accuracy(params, test_images, test_labels)]]
    for epoch, images, labels in train.batches(rng, train_images, train_labels, use_augmentation):
        layers = train.forward(params, images)
        grads = train.backward(params, layers, labels)
        optimizer.step(params, grads, train.learning_rate(epoch))
        if optimizer.t in check_steps:
            seen = optimizer.t * train.BATCH_SIZE
            curve.append([seen, train.accuracy(params, test_images, test_labels)])

    guesses = train.predict(params, test_images)
    confusion = np.zeros((10, 10), int)    # rows = true digit, columns = guess
    np.add.at(confusion, (test_labels, guesses), 1)
    return {
        "seed": seed,
        "augmented": use_augmentation,
        "accuracy": round(float((guesses == test_labels).mean()), 4),
        "perDigit": [round(float(confusion[d, d] / confusion[d].sum()), 4) for d in range(10)],
        "curve": [[seen, round(acc, 4)] for seen, acc in curve],
        "confusion": confusion.tolist(),
    }


def main():
    mnist_dir = sys.argv[1] if len(sys.argv) > 1 else train.DEFAULT_MNIST_DIR
    data = train.load_mnist(mnist_dir)
    test_labels = data[3]

    runs = []
    for use_augmentation in (True, False):
        for seed in SEEDS:
            start = time.time()
            result = run(seed, use_augmentation, *data)
            runs.append(result)
            name = "augmented" if use_augmentation else "no augmentation"
            print(f"{name:16s} seed {seed}: {result['accuracy'] * 100:.2f}%  "
                  f"({time.time() - start:.0f}s)", flush=True)

    with open(os.path.join(train.DATA_DIR, "experiments.json"), "w") as f:
        json.dump({"seeds": SEEDS, "testImages": len(test_labels), "runs": runs}, f,
                  separators=(",", ":"))


if __name__ == "__main__":
    main()
