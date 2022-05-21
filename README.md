# NEST
Nonparametric Decomposition of Sparse Tensors

This is the implementation of the NEST-1 and NEST-2 sparse tensor decomposition algorithms from the paper _Nonparametric Decomposition of Sparse Tensors_, by Conor Tillinghast and Shandian Zhe @ The Thirty-eighth International Conference on Machine Learning (ICML), 2021.

MIT license


# System Requirements

All code were tested under python 3.6 and TensorFlow 1.12.0. We recommend creating a virtual environment and pip installing the `requirements.txt` file.

# Usage 

To run the algorithms run either `main/run_nest1_trial.py` or `main/run_nest2_trial.py` from the command line. Each requires the path to the training data, the path to the test data and the rank of the decomposition. The batch size, learning rate and number of epochs can be specified as optional arguments. For example: 

		ex) python main/run_nest2_trial.py "data/alog/train-fold-1.txt" "data/alog/test-fold-1.txt" 3 --nepoch 700 --lr 0.001
		

# Citation
If you use our code please cite our paper

		@article{ctill2021nest,
		title={Nonparametric Decomposition of Sparse Tensors}, 
		author={Tillinghast, Conor and Zhe, Shandian},
  		journal={The Thirty-eighth International Conference on Machine Learning},
  		year={2021}}
