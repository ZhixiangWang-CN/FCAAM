# Federated Cross-Center Adaptive Alternating Model (FCAAM)

A privacy-preserving, robust, and interpretable federated learning framework for predicting Radiation Pneumonitis (RP) in multi-institutional radiotherapy studies.


## 🔍 Project Background
Accurate prediction of Radiation Pneumonitis (RP) — a common and potentially severe complication of thoracic radiotherapy — is critical for personalized treatment planning. However, two key challenges hinder progress:

1. **Poor Model Generalization**: AI models trained on single-center data often fail to perform well across heterogeneous multi-center cohorts (temporal/spatial data shifts).  
2. **Data Privacy Barriers**: Strict privacy regulations and ethical concerns prevent direct sharing of sensitive patient data across institutions, limiting the size and diversity of training datasets.  

This project introduces **FCAAM**, a novel federated learning framework designed to address both challenges: achieving robust cross-center generalization while fully preserving patient privacy.


## 🌟 Core Features
- **Privacy-Preserving**: No raw patient data leaves institutions; only model updates are shared (with DP-SGD protection).  
- **Robust Generalization**: Stable performance across heterogeneous multi-center cohorts (temporal/spatial shifts).  
- **Interpretable**: Clinically relevant feature learning, with built-in visualization tools for model interpretability.  
- **Clinically Translatable**: Web-based deployment prototype for integration into radiotherapy workflow.


## 🛠️ Quick Start
### Prerequisites
- Python 3.8 or higher  
- PyTorch 1.10 or higher  
- Federated Learning Libraries: `PySyft` or `FedML` (required for federated training)  
- Medical Imaging Tools: `SimpleITK` (for DICOM file processing)  
- Other Dependencies: See `requirements.txt`


### Installation
1. Clone the repository  
   ```bash
   git clone https://github.com/[your-username]/FCAAM-RP-Prediction.git
   cd FCAAM-RP-Prediction
