# Spatial Transformer Netork (STN) implemented with Tensorflow 2 Keras
  The project includes:
  * Tensorflow docker container to run the code
  * Generation of distorted MNIST dataset
  * Full training pipeline and evaluation
  * Demonstration using STN on manually distorted MNIST dataset and cluttered MNIST dataset 

### Run the training pipeline in docker
```bash
docker build -t tfqq
```

Launch the notebook
```bash
docker run -it -v {PATH_TO_REPO}/:/tf/stn -p 8888:8888 -p 0.0.0.0:6006:6006 tfqq:latest
```

Launch the bash
```
docker run -it -v {PATH_TO_REPO}/:/tf/stn -p 8888:8888 -p 0.0.0.0:6006:6006 tfqq:latest /bin/bash 
cd tf/stn
python prepare_distorted_dataset.py
python spatial_transformer_network_demo.py
```


### Experiment Results
|                        | Vanilla FC model trained on original MNIST | Vanilla FC model trained trained on both original MNIST and distorted MNIST | Vanilla FC model + Spatial Transfomer trained on both original MNIST and distorted MNIST |
|------------------------|--------------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Original MNIST         | 0.9783                                     | 0.9767                                                                      | 0.9867                                                                                   |
| Affine-distorted MNIST | 0.5180                                     | 0.7569                                                                      | 0.8928                                                                                   |
