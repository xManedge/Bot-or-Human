# Bot or Human Classification  

## Objective  
The primary goal of this project is to develop a machine learning model to classify entities as either "Bot" or "Human" based on provided features. This has applications in social media moderation, detecting automated bots, or chatbot identification.  

## Dataset Description  
The project uses the **DAIGT v2 Train Dataset** from Kaggle, which can be found [here](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset). This dataset includes labeled data to enable supervised learning.  

### Key Features of the Dataset  
- **Source:** DAIGT v2 (Dynamic AI-Guided Training Dataset).  
- **Attributes:** Features representing behavioral patterns, metadata, or text-like data indicative of bots or humans.  
- **Format:** Tabular format with feature columns and a target label.  

## Model Descriptions  
This project employs a BERT-based model for generating the embeddings, leveraging pre-trained transformer models for contextual understanding of text data.

About BERT
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art model developed by Google. It uses a transformer architecture to understand context in text by processing it in both directions (left-to-right and right-to-left). BERT is highly effective for natural language processing tasks, including classification.

In this project:

A pre-trained BERT model from the Hugging Face library is used to generate embeddings of each of the paragraphs.
The model benefits from transfer learning, which allows it to leverage vast prior training on general text corpora while adapting to the specific task.

### BiLSTM Model  
The embeddings from the BERT model is fed to the Bi LSTM model 
#### Model Structure    
- **BiLSTM Layer:** Processes input sequences in both forward and backward directions, capturing context from both past (left-to-right) and future (right-to-left) dependencies.  
- **Fully Connected Layer:** A dense layer that transforms the BiLSTM output to a lower-dimensional representation for classification.  
- **Output Layer:** A softmax or sigmoid activation for binary classification.  

#### Key Hyperparameters  
- **Embedding Dimension:** Determines the size of word/feature embeddings.
- **Hidden Units:** Number of LSTM units in each direction.  
- **Dropout Rate:** Regularization to prevent overfitting.  
- **Learning Rate:** Initial learning rate for the optimizer.  
- **Batch Size:** Number of samples per training step.  
- **Epochs:** Number of training iterations over the entire dataset.  


## Repository Contents  
- `Bot Or Human.ipynb`: A Jupyter Notebook containing the following:  
  - Data preprocessing and exploration.  
  - Implementation of both BiLSTM and BERT-based models using PyTorch.  
  - Model training, evaluation, and visualization of results.  

## Requirements  

### Prerequisites  
Ensure the following libraries are installed:  
- Python 3.x  
- Jupyter Notebook  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib/Seaborn (for visualization)  
- **PyTorch**: For deep learning implementation.  
- **Hugging Face Transformers**: For BERT.  

Install the necessary packages using the following command:  
```bash  
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers  
```  

### Dataset  
To access the dataset, download it from [Kaggle](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset) and place it in the project directory under `data/` or modify the file path in the notebook accordingly.  

## How to Use  

1. Clone this repository:  
   ```bash  
   git clone https://github.com/yourusername/bot-or-human.git  
   ```  
2. Navigate to the project directory and open the Jupyter Notebook:  
   ```bash  
   cd bot-or-human  
   jupyter notebook  
   ```  
3. Run the `Bot Or Human.ipynb` notebook to:  
   - Preprocess the data.  
   - Train and evaluate both BiLSTM and BERT-based models.  
   - Compare their performances.  

## Results  
The model showed tremdous performance with over 95% accuracy consistently in predicting the source of texts. All metrics of the model were outstanding and this could set a new standard for plagarism checking

## Future Work  
- Experiment with other transformer models like RoBERTa or GPT.  
- Investigate ensemble methods combining BiLSTM and BERT for better results.  
- Deploy the best-performing model in a real-world application.  

## Contributing  
Contributions are welcome! Please submit a pull request or open an issue for suggestions or improvements.  
