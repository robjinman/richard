Richard
=======

Named after one of the first programs I ever wrote as a child, Richard started out as a personal effort to learn more about machine learning. The original Richard was meant to be a "virus", but the most malicious thing I could do on my Psion Series 3 personal organiser was print the phrase "Richard is gaining power" in an infinite loop.

The new version of Richard is strictly benevolent.

In its current form, Richard is a CLI application that performs classification using a neural network. Supported layer types include dense, convolutional, and max pooling, but there will likely be others in the future.

GPU acceleration is supported with Vulkan compute shaders.


Building from source
--------------------

### Prerequisites

#### Linux

* cmake
* vcpkg
* Vulkan SDK (see https://vulkan.lunarg.com/sdk/home)

#### Windows

* cmake
* python 3
* Vulkan SDK

#### Mac OS

TODO

### Build

To build, just run the relevant workflow from the project root.

To see the list of workflows

```
    cmake --workflow --list-presets
```

For example, to make a debug build on linux

```
    cmake --workflow --preset=linux-debug
```

You can also run the configure/build steps separately

```
    cmake --preset=linux-debug
    cmake --build --preset=linux-debug
```


Usage
-----

To see usage

```
    ./richardcli/richardcli -h
```


Examples
--------

All examples are run from the build directory, e.g. build/linux/release, and assume you have datasets located under data/.

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
    ./richardcli/richardcli --train \
        --samples ../../../data/ocr/train.csv \
        --config ../../../data/ocr/config.json \
        --network ../../../data/ocr/network \
        --gpu

    ../richardcli/richardcli --eval \
        --samples ../../../data/ocr/test.csv \
        --network ../../../data/ocr/network \
        --gpu
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
    ./richardcli/richardcli --train \
        --samples ../../../data/catdog/train \
        --config ../../../data/catdog/config.json \
        --network ../../../data/catdog/network \
        --gpu

    ./richardcli/richardcli --eval \
        --samples ../../../data/catdog/test \
        --network ../../../data/catdog/network \
        --gpu
```


Profiling
---------

### CPU profile (Linux)

Install google perftools

```
    sudo apt install google-perftools
```

Build the linux-cpuprof preset

```
    cmake --workflow --preset=linux-cpuprof
```

Specify the intermediate file in the CPUPROFILE environment variable and run as usual, e.g.

```
    CPUPROFILE=./prof.out ./richardcli/richardcli --train \
        --samples ../../../data/ocr/train.csv \
        --config ../../../data/ocr/config_cnn.json \
        --network ../../../data/ocr/network
```

For text output

```
    google-pprof --text ./richardcli/richardcli ./prof.out > ./prof.txt
```

For graphical output

```
    google-pprof --gv ./richardcli/richardcli ./prof.out 
```

#### Interpreting the results

The text file should contain something like this

```
    Total: 2823 samples
        1166  41.3%  41.3%     1277  45.2% richard::computeCrossCorrelation
        1039  36.8%  78.1%     1145  40.6% richard::computeFullCrossCorrelation
        199   7.0%  85.2%      199   7.0% richard::Kernel::at (inline)
        ...
```

The first column is the number of samples spent inside the function.

The second column is this same number expressed as a percentage of the total samples taken. So in this case, we spent 41.3% of the time executing computeCrossCorrelation.

The third column is the cumulative time spent inside the function. In this example, 85.2% of the execution time is accounted for by these top 3 functions.

The next two columns tell us for how long the given function was part of the call stack. In other words, it includes time spent executing child calls. So in this example, we spent 40.6% of the time inside computeFullCrossCorrelation (including child calls), but only 36.8% actually within the computeFullCrossCorrelation function itself.
