# README

This repository contains code and notebooks for conducting class incremental learning experiments on the CIFAR-10 and CIFAR-100 datasets using MLP (Multi-Layer Perceptron) and RNN (Recurrent Neural Network) models. The experiments focus on obtaining Neural Tangent Kernel (NTK) matrices and tracking accuracies during the learning process. Below is a detailed explanation of the files and their purposes.

---

## File Descriptions

### 1. **Experiment Notebooks**
These notebooks are used to run the class incremental learning experiments on CIFAR-10 and CIFAR-100 datasets. They generate NTK matrices and accuracy metrics, which are saved as `.pkl` files.

- **`experiment-cifar10-epoch10.ipynb`**:  
  This notebook runs the class incremental learning experiment on the CIFAR-10 dataset using both MLP and RNN models. The results are saved in the corresponding `.pkl` files.

- **`experiment-cifar100-epoch10.ipynb`**:  
  This notebook runs the class incremental learning experiment on the CIFAR-100 dataset using both MLP and RNN models. The results are saved in the corresponding `.pkl` files.

---

### 2. **Generated `.pkl` Files**
These files contain the metrics (NTK matrices and accuracies) obtained from the experiments. They are saved in a serialized format for later analysis and visualization.

- **`mlp_cifar10_2_2_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-10 dataset.

- **`rnn_cifar10_2_2_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-10 dataset.

- **`mlp_cifar100_10_10_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-100 dataset.

- **`rnn_cifar100_10_10_10epo_cuda.pkl`**:  
  Contains the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-100 dataset.

---

### 3. **Plotting Notebooks**
These notebooks are used to visualize the metrics (NTK matrices and accuracies) saved in the `.pkl` files.

- **`plot_mlp_cifar10_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-10 dataset.

- **`plot_rnn_cifar10_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-10 dataset.

- **`plot_mlp_cifar100_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the MLP model trained on the CIFAR-100 dataset.

- **`plot_rnn_cifar100_10epoch.ipynb`**:  
  This notebook plots the metrics (NTK matrices and accuracies) for the RNN model trained on the CIFAR-100 dataset.

---

## How to Use

1. **Run Experiments**:  
   - Use the `experiment-cifar10-epoch10.ipynb` and `experiment-cifar100-epoch10.ipynb` notebooks to run the class incremental learning experiments on CIFAR-10 and CIFAR-100 datasets, respectively.  
   - The results will be saved as `.pkl` files.

2. **Visualize Results**:  
   - Use the corresponding plotting notebooks (e.g., `plot_mlp_cifar10_10epoch.ipynb`) to load the `.pkl` files and generate visualizations of the NTK matrices and accuracies.

