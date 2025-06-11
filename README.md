# Brain Tumor Detection Using Deep Learning

A comprehensive deep learning solution for automated brain tumor detection from MRI scans using Convolutional Neural Networks (CNN). This project implements state-of-the-art computer vision techniques to assist medical professionals in early and accurate diagnosis of brain tumors.

## 🧠 Project Overview

Brain tumors are among the most serious medical conditions requiring early detection for effective treatment. This project leverages the power of deep learning to analyze brain MRI images and classify them as either containing a tumor or being tumor-free. The solution aims to provide a reliable, automated screening tool that can support medical diagnosis workflows.

### Why This Matters
- **Early Detection**: Enables faster identification of potential brain tumors
- **Medical Support**: Assists radiologists and medical professionals in diagnosis
- **Accessibility**: Provides automated screening capabilities where medical expertise may be limited
- **Accuracy**: Achieves high precision in tumor classification

## ✨ Key Features

- **High Accuracy Classification**: Achieves 88%+ accuracy in tumor detection
- **Robust Data Processing**: Comprehensive preprocessing pipeline for MRI images
- **Data Augmentation**: Advanced techniques to handle limited dataset sizes
- **Model Optimization**: Custom CNN architecture designed for medical imaging
- **Real-time Prediction**: Fast inference for clinical applications
- **Visualization Tools**: Clear result presentation and model interpretability

## 🛠️ Technical Stack

- **Deep Learning**: TensorFlow & Keras
- **Image Processing**: OpenCV, PIL
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Development**: Python 3.7+, Jupyter Notebook

## 📊 Dataset Information

**Source**: Brain MRI Images for Brain Tumor Detection (Kaggle)
- **Total Images**: 253 original MRI scans
- **Positive Cases**: 155 images with tumors
- **Negative Cases**: 98 images without tumors
- **Augmented Dataset**: Expanded to 2,065 images for robust training

### Data Distribution
| Category | Original | After Augmentation |
|----------|----------|-------------------|
| Tumor Present | 155 | 1,085 |
| No Tumor | 98 | 980 |
| **Total** | **253** | **2,065** |

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7 or higher
TensorFlow 2.x
OpenCV
NumPy
Matplotlib
Jupyter Notebook
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aaryan-549/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

2. **Install dependencies**
```bash
pip install tensorflow opencv-python numpy matplotlib jupyter pandas seaborn pillow
```

3. **Download the dataset**
```bash
# Dataset will be automatically organized into appropriate folders
# Original images in 'yes' and 'no' folders
# Augmented images in 'augmented_data' folder
```

### Quick Start

1. **Load and preprocess data**
```python
from src.data_preprocessing import load_and_preprocess_data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
```

2. **Train the model**
```python
from src.model import create_brain_tumor_model
model = create_brain_tumor_model()
model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

3. **Make predictions**
```python
from src.predict import predict_tumor
result = predict_tumor('path/to/mri_image.jpg')
print(f"Prediction: {'Tumor Detected' if result > 0.5 else 'No Tumor'}")
```

## 🏗️ Model Architecture

### Custom CNN Design

Our model uses a carefully designed architecture optimized for brain MRI analysis:

```
Input Layer (240, 240, 3)
    ↓
Zero Padding (2, 2)
    ↓
Conv2D (32 filters, 7x7 kernel)
    ↓
Batch Normalization
    ↓
ReLU Activation
    ↓
MaxPooling2D (4x4)
    ↓
MaxPooling2D (4x4)
    ↓
Flatten
    ↓
Dense (1 neuron, sigmoid)
    ↓
Output (Binary Classification)
```

### Design Rationale
- **Simplicity**: Optimized for the dataset size to prevent overfitting
- **Efficiency**: Balanced between accuracy and computational requirements
- **Robustness**: Includes batch normalization for stable training
- **Medical Focus**: Architecture tailored for medical imaging characteristics

## 📈 Model Performance

### Training Results
- **Best Validation Accuracy**: 91% (achieved at epoch 23)
- **Test Accuracy**: 88.7%
- **F1 Score**: 0.88
- **Training Duration**: 24 epochs

### Performance Metrics

| Dataset | Accuracy | F1 Score | Precision | Recall |
|---------|----------|----------|-----------|--------|
| Validation | 91% | 0.91 | 0.89 | 0.93 |
| Test | 89% | 0.88 | 0.87 | 0.89 |

### Learning Curves
The model shows excellent convergence with minimal overfitting, indicating optimal architecture choice for the given dataset size.

## 🔧 Data Processing Pipeline

### 1. Image Preprocessing
```python
def preprocess_image(image_path):
    # Load and crop brain region
    image = crop_brain_region(image_path)
    
    # Resize to standard dimensions
    image = resize_image(image, target_size=(240, 240))
    
    # Normalize pixel values
    image = normalize_pixels(image)
    
    return image
```

### 2. Data Augmentation
To address the limited dataset size and class imbalance:
- **Rotation**: ±15 degrees
- **Width/Height Shift**: ±10%
- **Zoom**: ±10%
- **Horizontal Flip**: Applied selectively
- **Brightness Adjustment**: ±20%

### 3. Data Splitting
- **Training**: 70% (1,445 images)
- **Validation**: 15% (310 images)
- **Testing**: 15% (310 images)

## 📁 Project Structure

```
Brain-Tumor-Detection/
│
├── data/
│   ├── yes/                    # Tumor-positive MRI images
│   ├── no/                     # Tumor-negative MRI images
│   └── augmented_data/         # Augmented dataset
│
├── models/
│   ├── best_model.h5          # Best performing model
│   └── checkpoints/           # Training checkpoints
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_augmentation.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── model.py               # Model architecture
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Model evaluation
│   └── predict.py             # Prediction utilities
│
├── results/
│   ├── training_plots/        # Loss and accuracy curves
│   ├── confusion_matrix.png   # Model performance visualization
│   └── sample_predictions/    # Example predictions
│
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🎯 Usage Examples

### Single Image Prediction
```python
import tensorflow as tf
from src.predict import load_model, predict_single_image

# Load trained model
model = load_model('models/best_model.h5')

# Predict on new image
image_path = 'path/to/new_mri.jpg'
prediction, confidence = predict_single_image(model, image_path)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

### Batch Prediction
```python
from src.predict import predict_batch

image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
predictions = predict_batch(model, image_paths)

for path, pred in zip(image_paths, predictions):
    print(f"{path}: {'Tumor' if pred > 0.5 else 'No Tumor'}")
```

### Model Evaluation
```python
from src.evaluate import evaluate_model

# Comprehensive evaluation
metrics = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

## 🔬 Research and Development

### Methodology
1. **Literature Review**: Studied existing approaches for medical image classification
2. **Architecture Design**: Developed custom CNN optimized for brain MRI analysis
3. **Hyperparameter Tuning**: Systematic optimization of learning parameters
4. **Validation Strategy**: Rigorous testing to ensure clinical applicability

### Future Enhancements
- [ ] Multi-class tumor type classification
- [ ] Integration with DICOM medical imaging standards
- [ ] Web-based deployment for clinical use
- [ ] Ensemble methods for improved accuracy
- [ ] Explainable AI features for medical interpretability

## 🧪 Experimental Results

### Model Comparison
We tested various architectures before settling on our custom design:

| Model Type | Accuracy | Training Time | Memory Usage |
|------------|----------|---------------|--------------|
| ResNet50 (Transfer) | 85% | 45 min | High |
| VGG16 (Transfer) | 82% | 38 min | High |
| **Custom CNN** | **89%** | **15 min** | **Low** |

### Key Findings
- Transfer learning models were prone to overfitting on this dataset
- Simpler architectures performed better with limited data
- Data augmentation was crucial for achieving good generalization

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/improvement`)
3. **Make your changes** and add tests
4. **Commit your changes** (`git commit -am 'Add new feature'`)
5. **Push to the branch** (`git push origin feature/improvement`)
6. **Create a Pull Request**

### Contribution Areas
- Model architecture improvements
- Additional preprocessing techniques
- Performance optimization
- Documentation enhancements
- Testing and validation

## 📋 Requirements

```
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
pillow>=8.3.0
jupyter>=1.0.0
scikit-learn>=1.0.0
```

## ⚠️ Important Notes

### Medical Disclaimer
This tool is designed for educational and research purposes. It should not be used as a substitute for professional medical diagnosis. Always consult qualified medical professionals for actual medical decisions.

### Dataset Limitations
- Limited dataset size may affect generalization
- Images are from specific MRI protocols
- Model performance may vary with different imaging equipment

## 📚 References and Acknowledgments

- **Dataset Source**: [Brain MRI Images for Brain Tumor Detection - Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Deep Learning Framework**: TensorFlow and Keras teams
- **Medical Imaging Research**: Various publications on CNN applications in medical diagnosis

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project demonstrates the application of deep learning in medical imaging and serves as a foundation for further research in automated medical diagnosis.*
