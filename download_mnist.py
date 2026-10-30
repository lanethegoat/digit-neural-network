"""
Downloads the MNIST handwritten digit dataset (the data this network was
trained and tested on) into training/mnist/.

Run:  python3 download_mnist.py
Needs only Python 3, no extra packages. About 11 MB.
"""
import os, urllib.request

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",   # 60,000 training pictures
    "train-labels-idx1-ubyte.gz",   # the answer (0-9) for each training picture
    "t10k-images-idx3-ubyte.gz",    # 10,000 test pictures
    "t10k-labels-idx1-ubyte.gz",    # the answer for each test picture
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "mnist")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            print("already have", name)
            continue
        print("downloading", name)
        urllib.request.urlretrieve(MIRROR + name, path)
    print("done, files are in", OUT)


if __name__ == "__main__":
    main()
