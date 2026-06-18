# Autism-Spectrum-Disorder-ASD-Classification-Using-a-Neural-Network
This project implements a supervised machine learning pipeline for predicting Autism Spectrum Disorder (ASD) classification using participant screening and demographic data. The workflow includes data preprocessing, categorical feature encoding, neural network model development with Keras, model evaluation, and automated report generation. The model is trained as a binary classifier to distinguish between ASD-positive and ASD-negative cases based on questionnaire responses and associated attributes.

**Neural Network Architecture**

### Model Structure
A feedforward neural network is implemented using Keras.
```bash
model = Sequential()
```

**Expected Output**

After execution, the project produces:
1.	Trained neural network model 
2.	Classification predictions 
3.	Accuracy score 
4.	Precision, Recall, and F1-score metrics 
5.	report.txt containing experiment results 
6.	Application logs for monitoring and troubleshooting

