***

# TOA-using-generative-predictive-models

## Technological Impact-Guided Technology Opportunity Analysis using a Generative–Predictive Machine Learning Model

This repository contains the source code, scripts, and configuration files necessary to reproduce the empirical analysis presented in the journal article, **"Technological impact-guided technology opportunity analysis using a generative–predictive machine learning model."** The proposed analytical framework is designed to identify novel technological domains that maximize the potential impact of existing, proven technologies.

### 📝 Paper Title and Link

This project ensures the **transparency and reproducibility** of the research findings, which have been deemed worthy of publication after revision by *Scientometrics*.

*   **Title:** **Technological impact-guided technology opportunity analysis using a generative–predictive machine learning model**.
*   **Authors:** Changyong (and co-authors).
*   **Manuscript Version:** Revised manuscript (20250903).pdf.
*   **Journal:** *Scientometrics*.
*   **DOI/Link:** (The final DOI link will be added here upon journal publication.)

### 🌟 Project Overview and Objective

#### Overview
Traditional Technology Opportunity Analysis (TOA) approaches primarily focus on forecasting entirely new or nascent technologies. Our research, however, addresses the challenge of maximizing the potential of **existing, proven technologies** by identifying optimal domain-shift opportunities. We recognize that a technology's impact is highly dependent on its application domain, and **repurposing** can lead to significant breakthroughs (e.g., the Autographer case).

#### Objective
The goal is to develop a methodology based on an **integrated Generative–Predictive Machine Learning Model** (VAE and MLP architectures) to systematically discover new application domains in which existing technologies are expected to yield **greater technological impact** (L1 probability). The overall workflow is illustrated in **Fig. 1** of the manuscript.

### 🛠 Installation and Environment Setup

This project uses **PyTorch-based Python code** for data processing, model construction, and optimization.

#### 1. Prerequisites
Ensure you have Python installed. The required dependencies (based on the project scope and the ABCal reference) are listed in the presumed `requirements.txt` file, including PyTorch, NumPy, Pandas, Scikit-learn, and the Transformers library.

#### 2. Repository Cloning
```bash
# Clone the repository
git clone https://github.com/glee2/TOA-using-generative-predictive-models.git
cd TOA-using-generative-predictive-models

# Install dependencies
# [NOTE: This assumes a standard requirements.txt file exists]
pip install -r requirements.txt
```

### 💾 Data Download and Pre-processing Methods

The empirical analysis utilizes patent data sourced from the **PatentsView database** developed by the USPTO. The primary case study covers **133,654 patents related to Artificial Intelligence (AI) technology** granted between 2006 and 2020.

#### 1. Data Sources and Information Utilized
The framework relies on three critical information sources from patent documents:

*   **Technological Domains:** Cooperative Patent Classification (CPC) main group codes.
*   **Technological Functions:** Patent claim texts (specifically, independent claims).
*   **Technological Impact:** Forward citation counts received within five years of grant approval.

#### 2. Data Pre-processing Steps
1.  **Patent Collection:** Patent numbers are collected (e.g., from the AIPD dataset for AI patents) and corresponding HTML documents are retrieved from PatentsView.
2.  **Tokenization:** CPC codes are tokenized as variable-length sequences, and independent claim texts are tokenized using a WordPiece tokenizer.
3.  **Impact Labeling:** Forward citation counts are transformed into binary labels (L1/L2) based on the percentile rank (top 10%). For the AI dataset, patents with **more than 15 citations** were labeled L1 (Breakthrough).
4.  **Splitting:** The dataset is split using stratified random sampling into Training (70%), Validation (20%), and Test (10%) sets.

*   **Required Data Table:** Example of the collected data structure is presented in **Table 3** of the manuscript.

### 🧠 Model Training and Evaluation Methods

The core of the framework is the **generative–predictive integrated machine learning model**.

#### 1. Model Architecture and Joint Training
The model integrates two components (illustrated in **Fig. 2**):
*   **Generative Component (VAE):** Uses RNNs and a Transformer encoder to map patent classes and claims into a compressed **continuous latent vector ($z$)** and reconstruct the original CPC sequence. This process constructs the **impact-centric technology landscape**.
*   **Predictive Component (MLP):** Takes the latent vector $z$ as input to predict the binary technological impact label (L1/L2 probability).

The VAE and MLP are **jointly trained** using an integrated loss function that combines reconstruction, regularization (KL divergence), and prediction losses. A scaling factor $\alpha=0.2$ is used to balance the loss scales.

*   **Required Configuration Table:** Hyperparameters used for model creation and training are listed in **Table A1** (Appendix A).

#### 2. Model Performance Evaluation
Evaluation is conducted on the test set (10% of the data).

*   **Generative Performance:** Measured by **Jaccard Similarity** between the original and generated CPC sequences. The overall average for the AI analysis was approximately 0.7493.
*   **Predictive Performance:** Evaluated using Accuracy, Precision, Recall, and F1-score. For the L1 label, the **Recall value was much higher than the Precision value**, suggesting the model is "aggressive in predicting L1 labels" and focuses on learning the distinct features of breakthrough technologies.

*   **Required Performance Table:** Performance evaluation results are summarized in **Table 7(a) and 7(b)**.

### 🔍 Technology Opportunity Search and Result Reproduction

Technology opportunities are identified by systematically exploring the continuous latent space using the gradient ascent search algorithm.

#### 1. Gradient Ascent Search Methodology
The search process aims to maximize the L1 probability (high technological impact):
*   **Objective Function ($f$):** The predicted L1 probability output by the MLP.
*   **Search Mechanism:** The initial latent vector ($p^*$) is iteratively updated along the direction of the steepest ascent (the gradient, $\nabla f(p^*)$) until the L1 probability exceeds a desired termination threshold (e.g., 0.8).
*   **Rationale:** Optimization is performed in the VAE's continuous latent space because the original CPC sequence space is discrete, preventing gradient-based continuous optimization. The joint training ensures the latent space is "impact-centric," increasing the likelihood that the newly identified domains are both realistic and high-impact.

*   **Required Figures/Tables:** The exploration process is conceptually illustrated in **Fig. 3**. Examples of domain shifts during exploration are provided in **Table 6**.

#### 2. Validation and Reproduction of Results

The feasibility and practicality of the identified opportunities are verified through two validation schemes. Reproduction requires running the model training, search, and subsequent validation scripts.

*   **Micro-Validation Analysis:** Compares the technological impact of subsequent patents citing the focal patent when they remain in the original domain ($P_{citing/remained}$) versus when they shift to the new domain identified by the model ($P_{citing/shifted}$).
    *   **Result:** $P_{citing/shifted}$ patents showed an average forward citation count of **16.9016**, significantly higher than $P_{citing/remained}$ (5.6662), and a higher proportion of breakthrough inventions (about 30%).
    *   **Required Figures/Tables:** The validation structure is shown in **Fig. 4**. Micro-validation results are found in **Table 8**. The effect of the termination threshold is detailed in **Table 9**.
*   **Macro-Validation Analysis:** Evaluates the collective influence on technology trends.
    *   **Result:** Patents in the new domains exhibited a higher overall average citation count (13.4855) than those in the original domains (10.3353), confirming a positive influence on technology trends.
    *   **Required Table:** Macro-validation results are in **Table 11**.
*   **Robustness Test:** The framework's applicability was confirmed in the **Semiconductor technology field** as well, showing similar validation patterns.

### 📜 Code Usage and Citation Guidelines

#### License
This code is released under an open-source license, such as the **Apache-2.0 License** (or MIT License, to be specified upon final release). Use, modification, and distribution are permitted under the terms of the license.

#### Code Availability Statement
The public release of this code ensures the **transparency, reproducibility, and advancement** of the research community.

#### Citation
If you use this code or the associated methodology in your research, please cite the following paper:

```bibtex
@article{Choi_Technological_impact_guided_2025,
  title={Technological impact-guided technology opportunity analysis using a generative--predictive machine learning model},
  author={Changyong, C. et al. [Full Author List Upon Publication]}, 
  journal={Scientometrics},
  year={2025}, 
  note={Revised manuscript (20250903) accepted for publication},
  doi={DOI to be added upon publication}
}
```