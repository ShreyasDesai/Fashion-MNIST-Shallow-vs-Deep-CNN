# Fashion-MNIST: Shallow vs Deep CNN Comparative Study

This repository contains a complete, graded assignment comparing the performance, architectural complexity, and generalization capacity of a **Shallow CNN** and a **Deep CNN** on the **Fashion-MNIST** dataset.

## Project Goal
To implement, train, evaluate, and compare two different Convolutional Neural Network (CNN) architectures on the same Fashion-MNIST dataset under identical training configurations. The comparative study analyzes performance (accuracies), efficiency (parameter counts, training times), and overfitting behaviors (learning curves).

---

## 📂 Repository Structure
```text
├── Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb  # Main Jupyter Notebook (fully executed)
├── Comparative_Report.md                     # Detailed scientific comparative report
├── cnn_comparison_table.csv                  # Extracted comparison stats in CSV
├── README.md                                 # Project documentation (this file)
├── submission_instructions.txt               # Help file for GitHub upload & Vlearn submission
├── fashion_mnist_classes.png                 # Sample images from the 10 classes
├── shallow_cnn_learning_curves.png           # Accuracy/Loss plots for Shallow CNN
├── deep_cnn_learning_curves.png             # Accuracy/Loss plots for Deep CNN
├── shallow_cnn_predictions.png              # Sample predictions for Shallow CNN
├── deep_cnn_predictions.png                 # Sample predictions for Deep CNN
└── confusion_matrices.png                    # Confusion matrix heatmaps for both models
```

---

## 🛠️ Setup & Installation

### 1. Requirements
Ensure you have Python 3.8+ installed (tested on Python 3.13.1). The required libraries are:
- `tensorflow` (>=2.16)
- `numpy`
- `matplotlib`
- `seaborn`
- `pandas`
- `scikit-learn`
- `jupyter` / `notebook`

### 2. Quick Install
Install all required libraries using `pip`:
```bash
pip install tensorflow matplotlib numpy pandas scikit-learn seaborn notebook
```

---

## 🚀 How to Run the Notebook
1. Clone or download this repository.
2. Open a terminal/command prompt in the project folder and start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open the `Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb` notebook.
4. Run the cells sequentially to observe data loading, visualization, training logs, plots, prediction analysis, and final conclusion.

---

## 📊 Summary of Results

Our experimental training run achieved the following results:

| Metric | Shallow CNN | Deep CNN |
| :--- | :---: | :---: |
| **Number of Conv Layers** | 2 | 4 |
| **Total Parameters** | 1,011,466 | 889,834 |
| **Training Accuracy** | 97.65% | 93.74% |
| **Validation Accuracy** | 92.13% | 92.85% |
| **Test Accuracy** | **91.84%** | **92.58%** |
| **Overfitting Observed?** | Yes (Gap: 5.52%) | Minimal (Gap: 0.89%) |
| **Training Time** | 180.72 s | 313.48 s |

### Key Findings:
- **Parameter Efficiency**: The **Deep CNN is 12% smaller in parameters** than the Shallow CNN despite having double the layers. This is due to progressive pooling layers reducing the spatial dimension ($5 \times 5$ vs $11 \times 11$), shrinking the size of the Dense layer inputs.
- **Overfitting & Generalization**: The Shallow CNN overfitted the training set (97.65% train vs 91.84% test). The Deep CNN showed negligible overfitting (93.74% train vs 92.58% test), proving that **Dropout** regularization works exceptionally well.
- **Error Analysis**: Hardest classes are **Shirt** (Class 6), **T-shirt** (Class 0), and **Pullover** (Class 2) due to visual overlaps. Easiest classes are **Trouser** (Class 1), **Bag** (Class 8), and **Ankle boot** (Class 9).
