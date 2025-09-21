[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://link-to-your-paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper: "Technological impact-guided technology opportunity analysis using a generative–predictive machine learning model".

> **Note**: Add paper link once published.

## Overview

### Project Background
Previous studies on identifying promising technology opportunities have effectively generated new technology ideas, but their practical application has been limited. Technology impact varies depending on application domains, suggesting that shifts in these domains can create new technological opportunities. This project addresses the need for methodologies to identify new application domains for existing technologies to enhance their impact and usability.

### Project Objective
We develop a methodology based on an integrated structure of machine learning generative and predictive models to discover new application domains where existing technologies are expected to have greater technological impact.

*[Figure: Insert conceptual framework diagram showing the overall methodology]*

## Model Architecture

Our model integrates a Variational Auto-Encoder (VAE) with a Multilayer Perceptron (MLP):
- The VAE consists of:
  - An RNN encoder that processes CPC codes (technology application domains)
  - A Transformer encoder that processes patent claim texts (technological functions)
  - An RNN decoder that reconstructs CPC code sets
- The MLP takes the latent vectors from the VAE and predicts technological impact levels
- Both components are jointly trained with an integrated loss function

*[Figure: Insert model architecture diagram showing the VAE-MLP structure]*

## Installation

### Requirements
```bash
# Clone this repository
git clone https://github.com/glee2/TOA-using-generative-predictive-models.git
cd TOA-using-generative-predictive-models

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The main dependencies include:
- PyTorch
- Transformers
- Accelerate
- NLTK
- Pandas
- NumPy
- Scikit-learn
- Optuna

## Data

### Data Source
We use USPTO patent data with the following components:
- Patent classification codes (CPC) → Technology application domains
- Patent claim texts → Technological functions
- Patent forward citations → Level of technological impact

### Data Preparation
```bash
# Download and preprocess the patent data
python src/collect_patents_v8.ipynb

# Prepare the dataset for model training
python src/data.py --config configs/configs.json
```

*[Figure: Insert data preprocessing pipeline diagram]*

## Training

### Configuration
The model's hyperparameters can be configured in the `configs/configs.json` file. We also provide pre-optimized configurations in the `configs/best_hparam/` directory.

### Model Training
```bash
# Train the model with default configurations
python src/main.py --config configs/configs.json

# For hyperparameter optimization
python src/main.py --config configs/configs.json --optimize
```

*[Figure: Insert training process diagram showing the joint training of VAE and MLP]*

## Evaluation

### Model Reliability Evaluation
- The generative performance is evaluated using Jaccard similarity
- The predictive performance is assessed using accuracy, precision, recall, and F1-score

### Practicality Evaluation
We evaluate the practicality of our model by verifying whether the domain shifts identified occur in reality and lead to changes in technological impact.

```bash
# Run model evaluation
python src/main.py --config configs/configs.json --evaluate
```

Alternatively, you can use the validation notebook:
```bash
jupyter notebook src/Validation_v9.51\(refactored\).ipynb
```

## Technology Opportunity Exploration

After training the model, we explore the technology landscape using a Gradient Ascent Search algorithm to identify promising application domains:

```bash
# Set parameters in the validation notebook
# - Load the trained model
# - Set L1_threshold, n_iter, and step_size
# - Run the exploration algorithm
jupyter notebook src/Validation_v9.51\(refactored\).ipynb
```

Key parameters:
- `L1_threshold`: Threshold for determining innovative technology (default: 0.8)
- `n_iter`: Number of iterations for gradient ascent (default: 30)
- `step_size`: Step size for gradient ascent (default: 40)

*[Figure: Insert technology landscape visualization showing exploration paths]*

## Result Reproduction

To reproduce the results from our paper:
1. Use the configurations in `configs/USED_configs/`
2. Train the model with these configurations
3. Run the validation notebook with the same parameters mentioned in the paper

## Citation

If you use this code or find our work useful for your research, please cite our paper:

```
@article{lee2025technological,
  title={Technological impact-guided technology opportunity analysis using a generative–predictive machine learning model},
  author={Lee, Gyumin and [Other Authors]},
  journal={[Journal Name]},
  year={2025},
  publisher={[Publisher]}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or comments, please open an issue or contact the authors directly:
- Gyumin Lee - [Contact Information]

glee2: README.md 파일을 마크다운 형식으로 만들어서 바로 사용할 수 있게 해줘.

GitHub Copilot: # Technological Impact-Guided Technology Opportunity Analysis Using a Generative-Predictive Machine Learning Model

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://github.com/glee2/TOA-using-generative-predictive-models/blob/main/Revised%20manuscript%20(20250903).pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper: "Technological impact-guided technology opportunity analysis using a generative–predictive machine learning model".

## Overview

### Project Background
Previous studies on identifying promising technology opportunities have effectively generated new technology ideas, but their practical application has been limited. Technology impact varies depending on application domains, suggesting that shifts in these domains can create new technological opportunities. This project addresses the need for methodologies to identify new application domains for existing technologies to enhance their impact and usability.

### Project Objective
We develop a methodology based on an integrated structure of machine learning generative and predictive models to discover new application domains where existing technologies are expected to have greater technological impact.

![Conceptual Framework](figures/conceptual_framework.png)
*Figure 1: Conceptual framework showing the overall methodology*

## Model Architecture

Our model integrates a Variational Auto-Encoder (VAE) with a Multilayer Perceptron (MLP):
- The VAE consists of:
  - An RNN encoder that processes CPC codes (technology application domains)
  - A Transformer encoder that processes patent claim texts (technological functions)
  - An RNN decoder that reconstructs CPC code sets
- The MLP takes the latent vectors from the VAE and predicts technological impact levels
- Both components are jointly trained with an integrated loss function

![Model Architecture](figures/model_architecture.png)
*Figure 2: Model architecture showing the VAE-MLP structure*

## Installation

### Requirements
```bash
# Clone this repository
git clone https://github.com/glee2/TOA-using-generative-predictive-models.git
cd TOA-using-generative-predictive-models

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The main dependencies include:
- PyTorch
- Transformers
- Accelerate
- NLTK
- Pandas
- NumPy
- Scikit-learn
- Optuna

## Data

### Data Source
We use USPTO patent data with the following components:
- Patent classification codes (CPC) → Technology application domains
- Patent claim texts → Technological functions
- Patent forward citations → Level of technological impact

### Data Preparation
```bash
# Download and preprocess the patent data
python src/collect_patents_v8.ipynb

# Prepare the dataset for model training
python src/data.py --config configs/configs.json
```

![Data Preprocessing](figures/data_preprocessing.png)
*Figure 3: Data preprocessing pipeline*

## Training

### Configuration
The model's hyperparameters can be configured in the `configs/configs.json` file. We also provide pre-optimized configurations in the `configs/best_hparam/` directory.

### Model Training
```bash
# Train the model with default configurations
python src/main.py --config configs/configs.json

# For hyperparameter optimization
python src/main.py --config configs/configs.json --do_tune
```

![Training Process](figures/training_process.png)
*Figure 4: Training process showing the joint training of VAE and MLP*

## Evaluation

### Model Reliability Evaluation
- The generative performance is evaluated using Jaccard similarity
- The predictive performance is assessed using accuracy, precision, recall, and F1-score

### Practicality Evaluation
We evaluate the practicality of our model by verifying whether the domain shifts identified occur in reality and lead to changes in technological impact.

```bash
# Run model evaluation
python src/main.py --config configs/configs.json --do_eval

# For evaluating on training set as well
python src/main.py --config configs/configs.json --do_eval --eval_train_set
```

Alternatively, you can use the validation notebook:
```bash
jupyter notebook src/Validation_v9.51\(refactored\).ipynb
```

## Technology Opportunity Exploration

After training the model, we explore the technology landscape using a Gradient Ascent Search algorithm to identify promising application domains:

```bash
# Set parameters in the validation notebook
# - Load the trained model
# - Set L1_threshold, n_iter, and step_size
# - Run the exploration algorithm
jupyter notebook src/Validation_v9.51\(refactored\).ipynb
```

Key parameters:
- `L1_threshold`: Threshold for determining innovative technology (default: 0.8)
- `n_iter`: Number of iterations for gradient ascent (default: 30)
- `step_size`: Step size for gradient ascent (default: 40)

![Technology Landscape](figures/technology_landscape.png)
*Figure 5: Technology landscape visualization showing exploration paths*

## Result Reproduction

To reproduce the results from our paper:
1. Use the configurations in `configs/USED_configs/`
2. Train the model with these configurations
3. Run the validation notebook with the same parameters mentioned in the paper

```bash
# Example command to reproduce results
python src/main.py --config configs/USED_configs/[CONFIGS]2025-09-03_1200.json
```

## Citation

If you use this code or find our work useful for your research, please cite our paper:

```
@article{lee2025technological,
  title={Technological impact-guided technology opportunity analysis using a generative–predictive machine learning model},
  author={Lee, Gyumin and others},
  journal={TBD},
  year={2025},
  publisher={TBD}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or comments, please open an issue or contact the authors directly:
- Gyumin Lee - glee2@github.com