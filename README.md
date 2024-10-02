# User Reviews Chatbot

![Chatbot Interface](https://drive.google.com/uc?export=view&id=1S87tZIsq3Tm9YgUcXdrXhWDMDefHnsxI)

This project is a chatbot designed to answer questions based on user reviews from Spotify. The chatbot uses FAISS for vector database creation and retrieval and is implemented using LangChain and Streamlit.

---

## Features

- Retrieves and answers questions based on user reviews from a large dataset.
- Provides quality scoring for the chatbot's responses based on relevance, accuracy, and clarity.
- Allows interaction through a Streamlit interface.

---

## Setup Instructions

### 1. Virtual Environment Setup

1. Navigate to the project directory.
2. Create and activate a virtual environment for the project. Use the following commands:

**For Windows:**
```bash
python -m venv Menv
Menv\Scripts\activate
```
**For macOS or Linux:**
```bash
python3 -m venv Menv
source Menv/bin/activate
```
3. Once the virtual environment is activated, install the dependencies:
```bash
pip install -r requirements.txt
```
### 2. Download the FAISS Index Files

The FAISS index files (`index.faiss` and `index.pkl`) are too large to upload to GitHub (2.84 GB, ~2 million data points). You can download them from this Google Drive link:

[Download FAISS Index Files](https://drive.google.com/drive/folders/1UuKuh_4QuWS4PJOuHiIUXZxWkz7uJvXQ?usp=sharing)

Place both the `index.faiss` and `index.pkl` files in the `faiss_index` directory within the project folder.

### 3. Download the Cleaned Dataset (Optional)

If you want to access the cleaned dataset (`cleaned_spotify_reviews.csv`), you can download it from the following Google Drive link:

[Download Cleaned Dataset](https://drive.google.com/drive/folders/1UuKuh_4QuWS4PJOuHiIUXZxWkz7uJvXQ?usp=sharing)

Place the dataset in an appropriate directory as needed (e.g., `Data Processing`).

### 4. OpenAI API Key Setup

You need an OpenAI API key to run the chatbot. Place your API key in the `secret_key.py` file. The structure of the file should look like this:

```python
openai_key = "your-openai-api-key"
```

If you don't have an OpenAI API key, feel free to contact me at: [calebeffendi.work@gmail.com](mailto:calebeffendi.work@gmail.com).

### 5. Run the Chatbot
After setting up the virtual environment, downloading the FAISS index, and placing the required files in the correct directories, you can now run the chatbot. Follow these steps:

1. Activate the virtual environment (if not already activated).
2. Run the Streamlit app:
```bash
pip install -r requirements.txt
```
Once the server is running, open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`) to interact with the chatbot.

### 6. Notes

1. Local Deployment Only: Due to the large size of the FAISS index files, this chatbot is designed to run on your local machine.
2. This project is not intended for deployment to a cloud platform like Streamlit due to file size constraints.
3. File Organization: Ensure that all files are organized as they are in the repository, and do not modify the file structure.

## Files Overview

1. `Data_Processing/`: Contains a Jupyter notebook for cleaning, processing, and preparing the data.
   - `Data_Cleaning.ipynb`: Jupyter notebook used for data cleaning.
   
2. `Demo/`: Contains the demo video of the chatbot project.
   - `Project Demo.mp4`

3. `faiss_index/`: This directory is where the FAISS index files (`index.faiss`, `index.pkl`) should be placed after downloading.

4. `Trial/`: Contains the notebook file related to creating the FAISS vector database and chatbot trials.
   - `Chatbot Trial.ipynb`: Jupyter notebook for creating the vector database and running chatbot trials.

5. `bot_qna.py`: This file contains the chatbot logic for question-answering based on the FAISS vector store.

6. `main.py`: Contains the Streamlit application code for the chatbot interface.

7. `requirements.txt`: Specifies the required Python packages to run the project.

8. `secret_key.py`: Placeholder for your API keys (ensure no sensitive keys are committed).

9. `README.md`: The file you are reading now.

## Project Directory Structure

The directory structure should look like this after you have set up everything:

```python
├── Data Processing/
│   └── Data_Cleaning.ipynb        # Jupyter notebook for cleaning, processing, and preparing the data
├── Demo/                          # Project demo video
│   └── Project Demo.mp4
├── faiss_index/                   # Faiss index files
│   ├── index.faiss
│   └── index.pkl
├── Trial/
│   └── Chatbot Trial.ipynb        # Jupyter notebook for creating the vector database and running chatbot trials
├── Menv/                          # Virtual environment directory
│   ├── etc/
│   ├── include/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/
│   └── pyenv.cfg
├── bot_qna.py                     # Chatbot logic for answering based on FAISS vector store
├── main.py                        # Streamlit app interface for the chatbot
├── requirements.txt               # Python dependencies
├── secret_key.py                  # Placeholder for API keys
└── README.md                      # Project overview and documentation
```


