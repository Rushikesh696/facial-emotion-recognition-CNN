# facial-emotion-recognition-CNN

This project focuses on classifying human emotions from facial images using Convolutional Neural Networks (CNN) and Transfer Learning techniques. It leverages deep learning to identify emotions such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise from grayscale facial images.

### Project Overview
Facial emotion recognition plays a key role in applications such as:

🎥 Human–Computer Interaction

🏫 E-learning engagement monitoring

🏥 Mental health analysis

🛡️ Security and surveillance

In this project, we compare and evaluate four different deep learning approaches:

Custom CNN Model from Scratch

Custom CNN with Data Augmentation

VGG16 (Transfer Learning)

ResNet50 (Transfer Learning)

### Dataset
Dataset: FER-2013 Emotion Dataset

Classes: angry, disgust, fear, happy, neutral, sad, surprise

Train Images: 22,968

Validation Images: 5,741

Test Images: 7,178

Image Size: 48×48 (grayscale)

 Model Architectures
**Custom CNN**
6 Convolutional Layers

Batch Normalization + Dropout

Dense Layer with 1024 neurons

Softmax Output for 7 classes

CNN with Data Augmentation
Same as Custom CNN

Added Image Augmentation: rotation, zoom, width & height shifts

**VGG16 (Transfer Learning)**
Pre-trained VGG16 base (without top layers)

Global Average Pooling

Dense Layers for classification

**ResNet50 (Transfer Learning)**
Pre-trained ResNet50V2

Fine-tuned with custom dense layers

Training Setup
Input Size: 48×48 grayscale

Batch Size: 64

Optimizer: Adam (lr = 0.0001)

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Callbacks:

ModelCheckpoint

EarlyStopping

ReduceLROnPlateau

CSVLogger

### Evaluation Metrics
✅ Training & Validation Accuracy/Loss plots

✅ Confusion Matrix (per-class performance)

✅ Classification Report (Precision, Recall, F1-score)

✅ Random Predictions Visualization (green = correct, red = incorrect)

### Conclusion
This project demonstrates how deep learning models, especially with transfer learning, can effectively recognize facial emotions. While the Custom CNN provided a good baseline, VGG16 and ResNet50 significantly boosted accuracy and generalization. With further fine-tuning and larger datasets, this approach can be used in real-world emotion detection applications.
