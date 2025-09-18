# TOA using generative and predictive models
Paper title: Technological impact-guided technology opportunity analysis using a generative–predictive machine learning model

### Project Background and Necessity
- Previous studies on identifying promising technology opportunities were effective in generating new technology ideas that do not yet exist, but their practical use was limited.  
- The impact of a technology varies depending on the fields where it is applied, suggesting that shifts in application domains can lead to new technological opportunities.  
- Therefore, to strengthen the technological impact and increase the usability of existing technologies, there is a need to develop a methodology for identifying new application domains for existing technologies.  

### Project Objective
- The goal is to develop a methodology that, based on an integrated structure of machine learning generative and predictive models, discovers new application domains in which existing technologies are expected to have greater technological impact.  

### Data – Patent Document Data (USPTO)
- Patent classification codes (CPC) → Technology application domains  
- Patent claim texts → Technological functions  
- Patent forward citations → Level of technological impact  
  - Based on the top 10% of forward citations within a technological field, patents are classified as L1 (innovative technology) or L2 (general technology).  

### Analysis Method – Generative–Predictive Integrated Model Structure
- Constructed by combining a VAE (Variational Auto-Encoder) and an MLP (Multilayer Perceptron).  
  - The VAE consists of: an RNN (Recurrent Neural Network) encoder that takes a set of CPC codes as input, a Transformer encoder that processes patent claim texts, and an RNN decoder that generates a CPC code set identical in structure to the input. The latent space formed through this process is regarded as the technology landscape.  
  - The MLP takes the latent vectors output by the VAE as input and predicts the technological impact level of the input technology.  
- The VAE and MLP are jointly trained using an integrated loss function, structuring the technology landscape so that technologies embedded in similar regions correspond to similar levels of technological impact.  

### Analysis Method – Technology Landscape Exploration
- After training the generative–predictive integrated model, the structured technology landscape is explored using a Gradient Ascent Search algorithm.  
- Through this search, new application domains where existing technologies are expected to show greater technological impact are identified as promising technology opportunities.  

### Performance Validation – Experimental Results
- **Model Reliability Evaluation**  
  - The performance of the model in generating CPC code sets is evaluated using Jaccard similarity.  
  - The predictive performance of technological impact levels is evaluated using Accuracy, Precision, Recall, and F1-score for binary classification.  

### Performance Validation – Practicality Evaluation
- **Model Practicality Evaluation**  
  - The practicality of the model is assessed by checking, based on actual patent citation relationships, whether the domain shifts identified by the model occur in reality and whether such shifts are accompanied by changes in technological impact.  