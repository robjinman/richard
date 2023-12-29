Richard
=======

**Richard is gaining power.**

Named after one of the first programs I ever wrote as a child, Richard started out as a personal effort to learn more about machine learning. In its current form, Richard is a CLI application that performs classification using a neural network. Supported layer types currently include dense, convolutional, and max pooling. Ultimately, I'd like this project to encompass a library, a CLI wrapper, a GUI application, and possibly bindings to other languages. GPU acceleration is supported with Vulkan compute shaders, but this is a work in progress.

The original Richard was meant to be a "virus", but the most malicious thing I could do on my Psion Series 3 personal organiser was print the phrase "Richard is gaining power" in an infinite loop.

This incarnation of Richard is purely benevolent.


Building
--------

### Linux

#### Prerequisites

Requires CMake.

Install development dependencies

```
    sudo apt install \
        build-essential \
        libboost-program-options-dev \
        libvulkan-dev \
        vulkan-validationLayers-dev
```

#### Compile

From the richard subdirectory, to make a release build, run

```
    cmake -B build/release -D CMAKE_BUILD_TYPE=Release
    cmake --build build/release
```

And for a debug build:

```
    cmake -B build/debug -D CMAKE_BUILD_TYPE=Debug
    cmake --build build/debug
```


Examples
--------

All examples are run from the build directory, e.g. richard/build/release.

### Classifying hand-written digits with a fully connected network

#### config.json

```
    {
        "data": {
            "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "shape": [784, 1, 1],
            "normalization": {
                "min": 0,
                "max": 255
            }
        },
        "dataLoader": {
          "fetchSize": 512
        },
        "classifier": {
            "network": {
                "hyperparams": {
                    "epochs": 30,
                    "batchSize": 1024,
                    "miniBatchSize": 32,
                },
                "hiddenLayers": [
                    {
                        "type": "dense",
                        "size": 320,
                        "learnRate": 0.1,
                        "learnRateDecay": 1.0,
                        "dropoutRate": 0.0
                    },
                    {
                        "type": "dense",
                        "size": 64,
                        "learnRate": 0.1,
                        "learnRateDecay": 1.0,
                        "dropoutRate": 0.0
                    }
                ],
                "outputLayer": {
                    "size": 10,
                    "learnRate": 0.1,
                    "learnRateDecay": 1.0
                }
            }
        }
    }

```

```
    ./richard --train \
        --samples ../../../data/ocr/train.csv \
        --config ../../../data/ocr/config.json \
        --network ../../../data/ocr/network \
        --gpu

    ./richard --eval \
        --samples ../../../data/ocr/test.csv \
        --network ../../../data/ocr/network
```

### Classifying cats and dogs with a CNN

#### config.json

```
    {
        "data": {
            "classes": ["cat", "dog"],
            "shape": [100, 100, 3],
            "normalization": {
                "min": 0,
                "max": 255
            }
        },
        "dataLoader": {
          "fetchSize": 512
        },
        "classifier": {
            "network": {
                "hyperparams": {
                    "epochs": 10,
                    "batchSize": 1024,
                    "miniBatchSize": 32,
                },
                "hiddenLayers": [
                    {
                        "type": "convolutional",
                        "depth": 32,
                        "kernelSize": [3, 3],
                        "learnRate": 0.01,
                        "learnRateDecay": 1.0,
                        "dropoutRate": 0.0
                    },
                    {
                        "type": "maxPooling",
                        "regionSize": [2, 2]
                    },
                    {
                        "type": "convolutional",
                        "depth": 64,
                        "kernelSize": [4, 4],
                        "learnRate": 0.01,
                        "learnRateDecay": 1.0,
                        "dropoutRate": 0.0
                    },
                    {
                        "type": "maxPooling",
                        "regionSize": [2, 2]
                    },
                    {
                        "type": "dense",
                        "size": 64,
                        "learnRate": 0.01,
                        "learnRateDecay": 1.0,
                        "dropoutRate": 0.0
                    }
                ],
                "outputLayer": {
                    "size": 2,
                    "learnRate": 0.01,
                    "learnRateDecay": 1.0
                }
            }
        }
    }
```

```
    ./richard --train \
        --samples ../../../data/catdog/train \
        --config ../../../data/catdog/config.json \
        --network ../../../data/catdog/network \
        --gpu

    ./richard --eval \
        --samples ../../../data/catdog/test \
        --network ../../../data/catdog/network
```

