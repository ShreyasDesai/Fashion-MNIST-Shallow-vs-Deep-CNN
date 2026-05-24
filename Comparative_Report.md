# Comparative Study of Shallow CNN vs Deep CNN on Fashion-MNIST
**Course Project**: Graded Assignment on Convolutional Neural Networks (CNN)  
**Author**: Shreyas (AI Student)  
**Date**: May 2026  

---

## 1. Executive Summary
This report presents a comprehensive comparative study between a **Shallow CNN** and a **Deep CNN** trained on the **Fashion-MNIST** dataset. The goal is to evaluate their classification performance, analyze model complexity, training efficiency, generalization capacity, and identify the strengths and weaknesses of each architecture.

Our experiments revealed that:
- The **Deep CNN** achieved a higher test accuracy of **92.58%** compared to **91.84%** for the Shallow CNN.
- Despite having more convolutional layers, the **Deep CNN is actually more parameter-efficient**, requiring **889,834 parameters** compared to **1,011,466 parameters** for the Shallow CNN. This parameter reduction is due to progressive spatial pooling.
- The **Deep CNN generalized significantly better** (overfitting gap of **0.89%**) compared to the Shallow CNN (overfitting gap of **5.52%**) due to the integration of **Dropout** regularization layers.

---

## 2. Dataset Overview
The **Fashion-MNIST** dataset consists of $28 \times 28$ grayscale images of fashion products across 10 classes:
- **0**: T-shirt/top
- **1**: Trouser
- **2**: Pullover
- **3**: Dress
- **4**: Coat
- **5**: Sandal
- **6**: Shirt
- **7**: Sneaker
- **8**: Bag
- **9**: Ankle boot

The dataset contains **60,000 training images** and **10,000 test images**. It serves as a direct, more challenging drop-in replacement for the original MNIST digit dataset.

---

## 3. Data Preprocessing & Rationale

Before feeding the images into the convolutional neural networks, two critical preprocessing steps were performed:

1. **Normalization**:
   - *Action*: Converted pixel values from integers in $[0, 255]$ to floats in $[0, 1]$ (by dividing by $255.0$).
   - *Rationale*: Normalization stabilizes and accelerates gradient descent. It ensures that input features have a similar scale, making the loss landscape more spherical and preventing gradients from exploding or vanishing. This leads to faster convergence and smoother training.
2. **Reshaping**:
   - *Action*: Reshaped images from shape $(N, 28, 28)$ to $(N, 28, 28, 1)$.
   - *Rationale*: Keras and TensorFlow 2D Convolution layers (`Conv2D`) require input tensors to have a 4D structure: `(batch_size, height, width, channels)`. Since Fashion-MNIST is grayscale, it only has a single color channel. Reshaping explicitly adds this channel dimension.

---

## 4. Model Architectures

### 4.1 Shallow CNN Architecture
The Shallow CNN is designed to be a quick, low-latency baseline. It features:
- **Conv2D Layer 1**: 32 filters ($3 \times 3$), ReLU activation. Output: $(26, 26, 32)$.
- **MaxPooling2D Layer 1**: $2 \times 2$ pool size. Output: $(13, 13, 32)$.
- **Conv2D Layer 2**: 64 filters ($3 \times 3$), ReLU activation. Output: $(11, 11, 64)$.
- **Flatten Layer**: Flattens features to a 1D vector of size $11 \times 11 \times 64 = 7,744$ nodes.
- **Dense Hidden Layer**: 128 units, ReLU activation.
- **Output Layer**: 10 units, Softmax activation.
- **Total Parameters**: **1,011,466**

### 4.2 Deep CNN Architecture
The Deep CNN is designed to extract a hierarchy of features and includes regularization to prevent overfitting:
- **Block 1**:
  - **Conv2D Layer 1**: 32 filters ($3 \times 3$, padding='same'), ReLU. Output: $(28, 28, 32)$.
  - **Conv2D Layer 2**: 32 filters ($3 \times 3$, padding='valid'), ReLU. Output: $(26, 26, 32)$.
  - **MaxPooling2D Layer 1**: $2 \times 2$ pool size. Output: $(13, 13, 32)$.
  - **Dropout**: 25% regularization.
- **Block 2**:
  - **Conv2D Layer 3**: 64 filters ($3 \times 3$, padding='same'), ReLU. Output: $(13, 13, 64)$.
  - **Conv2D Layer 4**: 64 filters ($3 \times 3$, padding='valid'), ReLU. Output: $(11, 11, 64)$.
  - **MaxPooling2D Layer 2**: $2 \times 2$ pool size. Output: $(5, 5, 64)$.
  - **Dropout**: 25% regularization.
- **Classification Head**:
  - **Flatten Layer**: Flattens features to a 1D vector of size $5 \times 5 \times 64 = 1,600$ nodes.
  - **Dense Hidden Layer**: 512 units, ReLU activation.
  - **Dropout**: 50% regularization.
  - **Output Layer**: 10 units, Softmax activation.
- **Total Parameters**: **889,834**

---

## 5. Experimental Results & Comparative Analysis

Below is the structured comparison of the two models based on our training run:

| Metric | Shallow CNN | Deep CNN | Analysis & Observations |
| :--- | :--- | :--- | :--- |
| **Number of Conv Layers** | 2 | 4 | Deep CNN uses double the feature extraction depth. |
| **Total Parameters** | 1,011,466 | 889,834 | **Deep CNN has 12% fewer parameters** due to progressive pooling. |
| **Training Accuracy** | 97.65% | 93.74% | Shallow CNN fits the training set more aggressively. |
| **Validation Accuracy** | 92.13% | 92.85% | Deep CNN generalizes better to unseen validation data. |
| **Test Accuracy** | **91.84%** | **92.58%** | **Deep CNN wins by 0.74% on the test set.** |
| **Overfitting Observed?** | **Yes (Gap: 5.52%)** | **Minimal (Gap: 0.89%)** | Shallow CNN overfits; Deep CNN remains highly regularized. |
| **Training Time** | 180.72 s (~3 mins) | 313.48 s (~5.2 mins) | Shallow CNN trains 1.7x faster. |

### 5.1 The Parameter-Efficiency Paradox
A key observation is that the **Deep CNN contains fewer total parameters (889,834)** than the **Shallow CNN (1,011,466)**, despite having double the number of convolutional layers. 
- *Why?* In the Shallow CNN, there is only one pooling layer. When the features are flattened, the vector size is **7,744**. Connecting this to a 128-node Dense layer requires $7,744 \times 128 = 991,232$ parameters (representing 98% of the model's total weight).
- In the Deep CNN, we use two pooling layers, which reduces the final spatial size to $5 \times 5$. The flattened vector is only **1,600** nodes. Connecting this to a larger 512-node Dense layer requires $1,600 \times 512 = 819,200$ weights. 
- This demonstrates that **strategic pooling is crucial for controlling model size** and computational complexity, allowing us to build deeper feature-extraction pipelines while reducing memory footprint.

### 5.2 Generalization and Overfitting
- The **Shallow CNN** shows clear signs of overfitting: its training accuracy rises to 97.65%, but its test accuracy stalls at 91.84% (a gap of 5.52%). Its validation loss curves start rising in later epochs.
- The **Deep CNN** achieves 93.74% training accuracy and 92.58% test accuracy (a tiny gap of 0.89%). This demonstrates outstanding generalization. The introduction of **Dropout (25% in convolutional layers, 50% in dense layer)** successfully forces the network to learn redundant representations, preventing reliance on specific co-adapted features.

---

## 6. Prediction and Error Analysis

### 6.1 Easiest Classes
Both models classify the following items with near-perfect accuracy (>97%):
- **Trouser** (Class 1)
- **Bag** (Class 8)
- **Ankle boot** (Class 9)
*Reason*: These classes possess highly distinct shapes and silhouettes. Trousers are vertically elongated; bags are boxy; boots have high ankles. They do not share significant visual features with other garments.

### 6.2 Hardest & Most Confused Classes
The models frequently confuse:
- **T-shirt/top** (Class 0) $\longleftrightarrow$ **Shirt** (Class 6)
- **Pullover** (Class 2) $\longleftrightarrow$ **Coat** (Class 4) $\longleftrightarrow$ **Shirt** (Class 6)
- **Sandal** (Class 5) $\longleftrightarrow$ **Sneaker** (Class 7)
*Reason*: These items share extremely similar contours, sleeve shapes, and textures. In a $28 \times 28$ grayscale image, it is difficult to distinguish a crew neck pullover from a collared shirt or a light summer coat, even for a human eye.

### 6.3 Deep CNN's Impact on Confusions
The Deep CNN reduced confusions between similar garments. By building deeper layers, the network learns to detect:
- Fine-grained features such as button lines, lapels, pocket seams (to distinguish **Coats** from **Pullovers**).
- Collar structures and button-down shirts vs simple crew-neck silhouettes (to distinguish **Shirts** from **T-shirts**).
This is reflected in the confusion matrices where the off-diagonal error numbers are significantly lower for the Deep CNN.

---

## 7. Conclusion and Recommendations

### 7.1 Responses to Key Questions
1. **Which model would you recommend for Fashion-MNIST?**
   - We recommend the **Deep CNN**. It delivers higher accuracy, is parameter-efficient, and generalizes exceptionally well, making it much more robust for practical deployments.
2. **Which model was more efficient?**
   - The **Shallow CNN** was more computationally efficient in terms of training time (3 minutes vs 5.2 minutes). However, the **Deep CNN** is more parameter-efficient (fewer parameters) and offers a much better balance between size and performance.
3. **Which model was more accurate?**
   - The **Deep CNN** was consistently more accurate, outperforming the Shallow CNN on test data (92.58% vs 91.84%).
4. **What did you learn from this comparative study?**
   - We learned that model depth must be paired with **regularization (Dropout)** and **pooling** to prevent overfitting and control model size. Adding layers without pooling leads to parameter explosions in dense layers, while adding layers without dropout leads to overfitting.

### 7.2 Summary of Deliverables
- **Jupyter Notebook**: [Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb) (fully executed, containing code, outputs, and explanations).
- **Class Visualizations**: [fashion_mnist_classes.png](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/fashion_mnist_classes.png)
- **Shallow CNN Learning Curves**: [shallow_cnn_learning_curves.png](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/shallow_cnn_learning_curves.png)
- **Deep CNN Learning Curves**: [deep_cnn_learning_curves.png](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/deep_cnn_learning_curves.png)
- **Shallow Predictions**: [shallow_cnn_predictions.png](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/shallow_cnn_predictions.png)
- **Deep Predictions**: [deep_cnn_predictions.png](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/deep_cnn_predictions.png)
- **Confusion Matrix Plot**: [confusion_matrices.png](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/confusion_matrices.png)
- **Comparison CSV Table**: [cnn_comparison_table.csv](file:///C:/Shreyas/antigravity/HeroVired/Graded%20Assignment%20on%20Convolutional%20Neural%20Networks%20(CNN)/cnn_comparison_table.csv)
