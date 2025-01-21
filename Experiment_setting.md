# Class Incremental Learning Experiment Setup

This document describes the setup and implementation details of our **class incremental learning (CIL)** experiment, including the dataset, model architecture, training process, NTK (Neural Tangent Kernel) computation, and accuracy testing.

---

## 1. **Dataset and Task Setup**

### Dataset
- **Dataset**: CIFAR-10
- **Preprocessing**:
  - Images are normalized using `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`.
  - Converted to tensors using `transforms.ToTensor()`.

### Class Incremental Learning (CIL) Setup
- **Incremental Task Configuration**:
  - Initial task: 2 classes.
  - Increment: 2 new classes added per task.
  - Total tasks: 5 (since CIFAR-10 has 10 classes).
- **Validation Split**:
  - Each task's dataset is split into 90% training and 10% validation using `split_train_val`.

---

## 2. **Model Architectures**

### MLP (Multi-Layer Perceptron)
<!-- - **Structure**: -->
- **Dimensions tested**:
    - `100`, `500`, `1000`, `5000`, `10000`.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss.

### RNN (Recurrent Neural Network)
- **Structure**:
  - Input layer: Flattened image (32x32x3 = 3072 dimensions).
  - Hidden state dimensions tested:
    - `100`, `500`, `1000`, `5000`, `10000`.
  - Output layer: Fully connected layer with 10 outputs (one per class).
- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss.

---

## 3. **Training Process**

### Epochs and Tasks
- **Epochs per Task**: 10.
- **Total Epochs**: 50 (5 tasks × 10 epochs per task).
- **Batch Size**: 32.

### Training and Validation
- For each task:
  - The model is trained on the current task's training data.
  - Validation is performed on the current task's validation data.
  - After each epoch, the model is evaluated on all previously seen tasks to measure **task accuracy**.

---

## 4. **NTK (Neural Tangent Kernel) Computation**

### NTK Calculation
- **Method**:
  - For a fixed set of samples (10 training and 10 validation samples per task, identical across different models), the NTK matrix is computed at the end of each epoch.
  - The NTK matrix is calculated using the gradients of the model's output with respect to its parameters:
    ```python
    ntk[i, j] = torch.dot(grads[i], grads[j])
    ```
  - Gradients are computed using `torch.autograd` and concatenated into a single vector.

### NTK Metrics
- **NTK Norm**: Frobenius norm of the NTK matrix.
- **Eigenvalues**: Maximum and minimum eigenvalues of the NTK matrix.

---

## 5. **Accuracy Testing**

### Task Accuracy
- After each epoch, the model is evaluated on the validation sets of all previously seen tasks.
- **Accuracy Calculation**:
  - The model's predictions are compared to the ground truth labels.
  - Accuracy is computed as:
    ```python
    accuracy = (correct_predictions) / (total_samples)
    ```

### Metrics Collected
- **Training Loss**: Loss on the current task's training data.
- **Validation Loss**: Loss on the current task's validation data.
- **Task Accuracies**: Accuracy on all previously seen tasks.

---

## 6. **Data Collection and Storage**

### Metrics Saved
- **NTK Matrices**:
  - NTK matrices on the predefined sample set from training data for each task.
  - NTK matrices on the predefined sample set from validation data for each task.
- **Task Accuracies**: Accuracy on all tasks after each epoch.
- **Model Configurations**: Metrics are saved for each model width (e.g., 100, 500, 1000, 5000, 10000).

### File Storage
- Metrics are saved in `.pkl` files:
  - `mlp_cifar10_2_2_10epo_cuda.pkl` for MLP models.
  - `rnn_cifar10_2_2_10epo_cuda.pkl` for RNN models.

---

## 7. **Visualization**

### Plots Generated
- **NTK Metrics**:
  - NTK norms, maximum eigenvalues, and minimum eigenvalues for both training and validation datasets.
- **Task Accuracies**:
  - Accuracy on each task's dataset across epochs.

### Plotting Function
- The `plot_ntk_metrics` function generates plots for NTK metrics and task accuracies, with x-axis labeled by task and epoch.

---

