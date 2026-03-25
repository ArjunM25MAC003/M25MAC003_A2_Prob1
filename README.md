# PROBLEM 1 IIT Jodhpur Word Embeddings Project

This project implements Word2Vec models (CBOW and Skip-gram) to learn word embeddings from textual data collected from IIT Jodhpur sources. It is part of the M25MAC003 assignment (Problem 1).

## Functionality

The main script `m25mac003_prob1.py` performs the following tasks:
1.  **Data Collection**:
    -   **Web Scraping**: Scrapes text from IIT Jodhpur websites (`iitj.ac.in` and mathematics department) and faculty pages using `requests` and `BeautifulSoup`.
    -   **PDF Extraction**: Extracts text from PDF files located in the `pdf/` directory using `pdfplumber`, download the upoloaded pdfs of academic regulations and research paper and save it in "pdf" folder to run.
2.  **Preprocessing**:
    -   Cleans text by removing dates, times, special characters, and extra spaces.
    -   Tokenizes text and removes stopwords (standard English + domain-specific: "shall", "may", "institute", "indian", "technology", "jodhpur", "iit", "also").
    -   Saves the cleaned corpus to `clean_corpus.txt`.
3.  **Model Training**:
    -   Trains a **CBOW (Continuous Bag of Words)** model using `gensim`.
    -   Trains a **Skip-gram** model using `gensim`.
    -   Saves trained models as `cbow_model.model` and `skipgram_model.model`.
4.  **Analysis & Visualization**:
    -   Evaluates semantic similarity.
    -   Visualizes word embeddings in 2D space using **t-SNE** (results saved in `Visualizations/`).

## Project Structure

-   `m25mac003_prob1.py`: The main script handling data collection, preprocessing, training, and visualization.
-   `pdf/`: Directory containing PDF documents (e.g., Academic Regulations) for text extraction.
-   `clean_corpus.txt`: The processed text data used for training.
-   `cbow_model.model`: Saved CBOW Word2Vec model.
-   `skipgram_model.model`: Saved Skip-gram Word2Vec model.
-   `Visualizations/`: Directory for generated plots.

## Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib nltk beautifulsoup4 scikit-learn gensim pdfplumber requests
```

## How to Run

1.  **Prerequisites**:
    *   Ensure you have an active internet connection (required for web scraping IIT Jodhpur websites).
    *   Install the dependencies listed above.

2.  **Prepare Data**:
    *   Create a folder named `pdf` in the project root if it doesn't exist.
    *   Place the PDF documents (e.g., Academic Regulations) you want to include in the corpus inside the `pdf/` folder.

3.  **Execute the Script**:
    Run the Python script from the terminal:

    ```bash
    python m25mac003_prob1.py
    ```

    **What happens next?**
    *   The script will scrape text from the websites.
    *   It will read and extract text from PDFs in the `pdf/` folder.
    *   It will clean and tokenize the data, saving it to `clean_corpus.txt`.
    *   It will train the CBOW and Skip-gram models, saving them as `.model` files.
    *   Finally, it will generate t-SNE visualization plots in the `Visualizations/` directory.

## Model Configuration
-   **CBOW**: Vector Size: 200, Window: 8, Negative Sampling: 10
-   **Skip-gram**: Vector Size: 100, Window: 5, Negative Sampling: 5

# PROBLEM 2-Character-Level Name Generation using RNN Variants

This project implements and compares three different Recurrent Neural Network (RNN) architectures for character-level name generation:
1.  **Vanilla RNN**
2.  **Bidirectional LSTM (BLSTM)**
3.  **RNN with Attention**

The objective is to train these models on a dataset of names and generate new, unique names. The models are evaluated based on training loss and qualitative metrics such as **Novelty Rate** and **Diversity**.

## Project Structure

-   `m25mac003_prob2.py`: The main Python script containing the implementation of the models, training loop, generation logic, and evaluation metrics.
-   `TrainingNames.txt`: A text file containing the dataset of names used for training.
-   `requirements.txt`: A list of Python dependencies required to run the project.

## Requirements

-   Python 3.x
-   PyTorch

## Installation

1.  **Clone or download** the repository to your local machine.
2.  **Navigate** to the project directory:
    ```bash
    cd path/to/M25MAC003_prob2
    ```
3.  **Install the required dependencies** using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure that the `TrainingNames.txt` file is located in the same directory as the script.
2.  Run the main script:
    ```bash
    python m25mac003_prob2.py
    ```

## How It Works

The script performs the following steps for each model:
1.  **Data Preparation**: Reads names from `TrainingNames.txt`, processes them (lowercasing), and builds a character vocabulary.
2.  **Model Initialization**: Initializes the specific RNN variant (Vanilla, BLSTM, or Attention).
3.  **Training**: Trains the model for a specified number of epochs (default is 20), printing the loss at each epoch.
4.  **Generation**: Generates a set of new names using the trained model.
5.  **Evaluation**: Calculates and prints:
    -   **Novelty Rate**: The proportion of generated names that are not in the training set.
    -   **Diversity**: The ratio of unique generated names to total generated names.

## Output

The script will output the training progress (loss per epoch) and evaluation metrics for each model variant directly to the console.

