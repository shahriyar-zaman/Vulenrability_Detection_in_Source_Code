
# Ensemble Stacking Approach of Large Language Models for Vulnerability Detection

<p align="center"> <strong>Official Implementation for IEEE BigData 2024 Submission</strong> </p>

## Abstract
This repository contains the official implementation of our paper *"An Ensemble Stacking Approach of Large Language Models for Enhanced Vulnerability Detection in Source Code"*. In this work, we propose an ensemble stacking approach that combines pre-trained models, such as CodeBERT, GraphCodeBERT, and UniXcoder, to improve vulnerability detection in source code. Our method fine-tunes these models on the Draper VDISC dataset and integrates their outputs using meta-classifiers such as Logistic Regression, SVM, Random Forest, and XGBoost, resulting in improved performance metrics like accuracy, precision, recall, F1-score, and AUC-score.

## Dataset Statistics
| Metric | CWE-119 (Memory) | CWE-120 (Buffer Overflow) | CWE-469 (Integer Overflow) | CWE-476 (Null Pointer) | CWE-other | Total |
|--------|------------------|---------------------------|----------------------------|-------------------------|-----------|-------|
| **Training Samples**   | 5,942  | 5,777  | 249   | 2,755  | 5,582  | 20,305 |
| **Validation Samples** | 1,142  | 1,099  | 53    | 535    | 1,071  | 3,900  |
| **Test Samples**       | 1,142  | 1,099  | 53    | 535    | 1,071  | 3,900  |



## Methodology
1. **Data Preprocessing:** We tokenize code snippets and apply downsampling to address class imbalance. Models are fine-tuned on the balanced dataset.
2. **Model Training:** CodeBERT, GraphCodeBERT, and UniXcoder are fine-tuned with vulnerability class labels.
3. **Meta-Classifier Stacking:** Outputs of the fine-tuned models are combined using meta-classifiers like Logistic Regression, SVM, Random Forest, and XGBoost to make the final prediction.
4. **Result Analysis:** Our ensemble stacking approach surpasses individual models in detecting vulnerabilities across multiple metrics.

![image](https://github.com/shahriyar-zaman/Vulenrability_Detection_in_Source_Code/blob/ba71f069da6e411992f1d47b3f83a38b8b04a451/Figures/vulnerability_methodology_figure.jpg)

**Fig. 1: Overall workflow diagram of our methodology**

## Results
The proposed ensemble stacking method improves the detection of code vulnerabilities significantly across all performance metrics. The combination of GraphCodeBERT and UniXcoder with an SVM meta-classifier achieves the best accuracy at 82.36%.

### Model Performances with baseline and ensemble stacking(ours)

| Model                       | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-Score |
|-----------------------------|--------------|---------------|------------|--------------|-----------|
| CodeBERT                     | 78.51        | 77.85         | 78.51      | 77.98        | 0.9216    |
| GraphCodeBERT                | 80.05        | 79.92         | 80.05      | 79.86        | 0.9336    |
| UniXcoder                    | 81.54        | 81.96         | 81.54      | 81.49        | 0.9380    |
| Ours (C+G) (SVM)             | 81.46        | 81.77         | 81.46      | 81.40        | 0.8996    |
| Ours (G+U) (SVM)             | 82.36        | 82.85         | 82.36      | 82.28        | 0.9053    |
| Ours (G+U) (LR)              | 82.36        | 82.59         | 82.36      | 82.21        | 0.9285    |

## Hyperparameters
### Transformer Models (CodeBERT, GraphCodeBERT, UniXcoder):
- **Batch Size:** 16
- **Epochs:** 10
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW
- **Max Token Length:** 512

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/shahriyar-zaman/Vulenrability_Detection_in_Source_Code.git
cd LLM_Vulnerability_Detection
```

### 2. Install Required Packages
Install the necessary libraries:
```bash
pip install transformers torch scikit-learn xgboost
```

### 3. Preprocess the Dataset
Run the preprocessing script:
```bash
python preprocess_data.py --input_file <input_data_file> --output_file <output_data_file>
```

## 🧪 Training and Test Splits
- **Training Samples:** 20,305
- **Validation Samples:** 3,900
- **Test Samples:** 3,900

You can modify the dataset splits as needed or experiment with other datasets.

## 📚 Model Training
To fine-tune a specific LLM, run:
```bash
python train_model.py --model codebert --epochs 10 --batch_size 16 --learning_rate 2e-5
```

To run the meta-classifier with stacking:
```bash
python train_meta_classifier.py --models codebert graphcodebert unixcoder --meta_classifier svm
```

## References
- Z. Feng et al., "CodeBERT: A Pre-trained Model for Programming and Natural Languages," 2020.
- D. Guo et al., "GraphCodeBERT: Pre-training Code Representations with Data Flow," 2020.
- D. Guo et al., "UniXcoder: Unified Cross-Modal Pre-training for Code Representation," 2022.
- L. Russell et al., "The Draper VDISC Dataset," 2018.

This README provides an overview of how to set up, train, and test the models for vulnerability detection using the proposed ensemble stacking approach.
