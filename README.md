![Logo](img/logo1.png "Logo")


**Try the app here:** [PDF Analyzer on Streamlit Cloud](https://mehmetba-pdf-analyze-streamlit-qa-app-5uufsy.streamlit.app/)

# PDF Analyzer App

PDF Analyzer App is a question-answering application that allows users to upload documents (PDF or TXT) and ask questions related to the content of those documents. The app utilizes various retrievers such as similarity search and support vector machines to provide relevant answers.

## Features

- Upload PDF or TXT documents
- Choose the type of retriever: Similarity Search or Support Vector Machines
- Generate sample question-answer pairs based on the uploaded documents
- Ask questions related to the content of the uploaded documents
- Get answers from the app using the selected retriever method

## Installation

Clone this repository:

```bash
git clone https://github.com/mehmetba/pdf-analyze-streamlit.git
cd pdf-analyze-streamlit
```

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
To run the app, simply execute the following command:

```bash
streamlit run qa_app.py
```

After running the command, you can access the app through your web browser using the provided URL.

Developed by [Mehmet Balioglu](https://twitter.com/mehmet_ba7)
