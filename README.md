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
python spatial_transformer_network_demo.py
```



