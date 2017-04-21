## Preparation
Run ```python mnist_convert_to_records.py --directory {MNIST_DIRECTORY}```
to prepare MNIST data for use with mnist_input.py

## Training
```python train.py --data_dir={DIRECTORY} --dataset=[minst|cifar10]```

See train.py for command line arguments

## Experiments
### CIFAR10 with 1 continuous dimension and 1 category 10 dimension
```python train.py --data_dir={DIRECTORY} --dataset=cifar10 --cont_dim=1 --discrete_dim=10 --iters=180000```

### CIFAR10 with 10 continuous dimension and 1 category 10 dimension
```python train.py --data_dir={DIRECTORY} --dataset=cifar10 --cont_dim=10 --discrete_dim=10 --iters=180000```

## Contributions
cifar10_input.py was modified from https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
mnist_input.py was modified from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py