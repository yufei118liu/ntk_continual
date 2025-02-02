# Analysis of Deep Neural Network Dynamics in Continual learning

This repository contains the code for a project done in the course Deep learning at ETH Zürich by

- Yufei Liu
- Yuzhi Liu
- Zirui Zhang
- Zixuan Chen

Our work provides a broad view of the evolution of the empirical NTK matrix throughout the training process, which enhances the understanding of how much the NTK theory aligns with the empirical results on finite width setting and facing drastic changes in input.

The repository contains code and notebooks for conducting class incremental learning experiments on the CIFAR-10 and CIFAR-100 datasets using MLP (Multi-Layer Perceptron) , RNN (Recurrent Neural Network) and Convolutional Neural Network (CNN) models. The experiments focus on obtaining Neural Tangent Kernel (NTK) matrices and tracking accuracies during the learning process. Below is a detailed explanation of the files and their purposes.

---

## File Descriptions

### 1. **Experiment Notebooks**
These notebooks are used to run the class incremental learning experiments on CIFAR-10 and CIFAR-100 datasets. They generate NTK matrices and accuracy metrics, which are saved as `.pkl` files.

- **`experiment-cifar10-epoch10.ipynb`**:  
  This notebook runs the class incremental learning experiment on the CIFAR-10 dataset using MLP, RNN and CNN models. The results are saved in the corresponding `.pkl` files.

- **`experiment-cifar100-epoch10.ipynb`**:  
  This notebook runs the class incremental learning experiment on the CIFAR-100 dataset using MLP, RNN and CNN models. The results are saved in the corresponding `.pkl` files.

- **`experiment-cifar10-epoch10_replay.ipynb`**:  
  This notebook runs the class incremental learning experiment with Experience Replay on the CIFAR-10 dataset using MLP and CNN models. The results are saved in the corresponding `.pkl` files.

- **`experiment-cifar100-epoch10_replay.ipynb`**:  
  This notebook runs the class incremental learning experiment with Experience Replay on the CIFAR-100 dataset using MLP and CNN models. The results are saved in the corresponding `.pkl` files.

---

### 2. **Generated `.pkl` Files**
These files contain the metrics (NTK matrices and accuracies) obtained from the experiments. They are saved in a serialized format for later analysis and visualization.

- **`mlp_cifar10_2_2_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-10 dataset.

- **`rnn_cifar10_2_2_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-10 dataset.

- **`cnn_cifar10_2_2_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the CNN model trained on the CIFAR-10 dataset.

- **`cnn_cifar10_replay_2_2_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the CNN model with Experience Replay trained on the CIFAR-10 dataset.

- **`mlp_cifar100_10_10_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-100 dataset.

- **`rnn_cifar100_10_10_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-100 dataset.
  
- **`cnn_cifar100_10_10_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the CNN model trained on the CIFAR-100 dataset.

- **`cnn_cifar100_replay_10_10_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the CNN model with Experience Replay trained on the CIFAR-100 dataset.

---

### 3. **Plotting Notebooks**
These notebooks are used to visualize the metrics (NTK matrices and accuracies) saved in the `.pkl` files.

- **`plot_mlp_cifar10_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-10 dataset.

- **`plot_rnn_cifar10_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-10 dataset.

- **`plot_cnn_cifar10_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the CNN model trained on the CIFAR-10 dataset.

- **`plot_mlp_cifar100_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-100 dataset.

- **`plot_rnn_cifar100_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-100 dataset.

- **`plot_cnn_cifar100_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the CNN model trained on the CIFAR-100 dataset.

---

## How to Use

1. **Run Experiments**:  
   - Use the `experiment-cifar10-epoch10.ipynb` and `experiment-cifar100-epoch10.ipynb` notebooks to run the class incremental learning experiments on CIFAR-10 and CIFAR-100 datasets, respectively.  
   - The results will be saved as `.pkl` files.

2. **Visualize Results**:  
   - Use the corresponding plotting notebooks (e.g., `plot_mlp_cifar10_10epoch.ipynb`) to load the `.pkl` files and generate visualizations of the NTK matrices and accuracies.

