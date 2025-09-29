# NeuroClassify: Brain Tumor Classification with Deep Learning

[![Model Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen.svg)](https://[LINK_TO_YOUR_NOTEBOOK_OR_RESULTS])

This project implements a deep learning solution for classifying brain tumors from MRI scans. Using a dataset containing four types of tumors (glioma, meningioma, pituitary) and images with no tumor, this project builds and trains an EfficientNetB0 model to achieve high diagnostic accuracy.

The entire workflow, from data loading and preprocessing to model training and performance evaluation, is documented within the Jupyter Notebook. The final model demonstrates excellent performance, achieving **98% accuracy** on the validation set.
---

## Project Pipeline

The project follows a systematic machine learning workflow:

1.  **Data Loading & Exploration:** The dataset is loaded from directories, and the distribution of the four classes is analyzed and visualized.
2.  **Image Preprocessing:** Images are resized to a uniform dimension (224x224 pixels) and normalized to scale pixel values between 0 and 1 for optimal model performance. No data augmentation is used in this project.
3.  **Model Architecture:** A pre-trained **EfficientNetB0** model is utilized as the base for transfer learning. Custom fully connected layers are added on top, including a `Dense` layer with ReLU activation and a final `Softmax` layer for multi-class classification.
4.  **Training & Validation:** The model is trained for 20 epochs using the Adam optimizer and `categorical_crossentropy` loss function. Performance is monitored on a separate validation set throughout the training process.
5.  **Performance Evaluation:** The trained model's performance is thoroughly evaluated using key metrics, including accuracy, a detailed classification report (precision, recall, F1-score), and a confusion matrix to visualize its predictive accuracy across different classes.

---

## Tech Stack

* **Core Libraries:** Python 3
* **Data Manipulation & Analysis:** NumPy
* **Deep Learning Framework:** TensorFlow, Keras
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning & Metrics:** Scikit-learn
* **Environment:** Jupyter Notebook, Google Colab

---

## Results & Evaluation

The model achieved outstanding results on the validation dataset after 20 epochs of training.

* **Validation Accuracy:** **98.28%**
* **Validation Loss:** 0.05

### Classification Report

The detailed report shows high precision, recall, and F1-scores across all four classes, indicating a well-balanced and robust model.

```
            precision    recall  f1-score   support

glioma           0.96      0.96      0.96       300
meningioma       0.96      0.99      0.97       306
no tumor         1.00      0.99      0.99       405
pituitary        1.00      0.98      0.99       300

accuracy                             0.98      1311
macro avg        0.98      0.98      0.98      1311
weighted avg     0.98      0.98      0.98      1311

````

### Confusion Matrix

The confusion matrix visually confirms the model's high accuracy, with the vast majority of predictions falling along the main diagonal, indicating correct classifications.

---

## How to Reproduce

To replicate the results of this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/NeuroClassify.git
    cd NeuroClassify
    ```

2.  **Download the dataset:**
    Download the "Brain Tumor MRI Dataset" from Kaggle and place it in the appropriate directory as referenced by the notebook.
    * [Link to Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

3.  **Open the Jupyter Notebook:**
    Launch Jupyter Notebook or open the `.ipynb` file in Google Colab.
    ```bash
    jupyter notebook NeuroClassify_modeltraining.ipynb
    ```

4.  **Run the cells:**
    Execute the cells sequentially to load the data, train the model, and view the evaluation results.
