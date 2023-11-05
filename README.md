Richard
=======

Richard is gaining power.

Examples
--------

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
          "fetchSize": 500
        },
        "classifier": {
            "network": {
                "hyperparams": {
                    "epochs": 30,
                    "maxBatchSize": 1000
                },
                "hiddenLayers": [
                    {
                        "type": "dense",
                        "size": 300,
                        "learnRate": 0.1,
                        "learnRateDecay": 1.0,
                        "dropoutRate": 0.0
                    },
                    {
                        "type": "dense",
                        "size": 80,
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
    ./richard --train --samples ../../data/ocr/train.csv --config ../../data/ocr/config.json --network ../../data/ocr/network
    ./richard --eval --samples ../../data/ocr/test.csv --network ../../data/ocr/network
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
          "fetchSize": 500
        },
        "classifier": {
            "network": {
                "hyperparams": {
                    "epochs": 10,
                    "maxBatchSize": 1000
                },
                "hiddenLayers": [
                    {
                        "type": "convolutional",
                        "depth": 32,
                        "kernelSize": [3, 3],
                        "learnRate": 0.01,
                        "learnRateDecay": 1.0
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
                        "learnRateDecay": 1.0
                    },
                    {
                        "type": "maxPooling",
                        "regionSize": [2, 2]
                    },
                    {
                        "type": "dense",
                        "size": 100,
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
    ./richard --train --samples ../../data/catdog/train --config ../../data/catdog/config.json --network ../../data/catdog/network
    ./richard --eval --samples ../../data/catdog/test --network ../../data/catdog/network
```

