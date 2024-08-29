# LegalChatbot

# PDF Question Answering Application

This project is a Streamlit-based application that extracts text from PDF documents and uses a language model to answer questions based on the extracted text.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/tr/chatwithpdfs.git
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Upload PDF documents through the Streamlit interface.

3. Ask questions based on the content of the uploaded PDFs.

## Functions

- **get_pdf_text(pdf_docs)**: Extracts text from a list of PDF documents.
    ```python
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        text = re.sub(r'\n', ' ', text)
        return text
    ```

- **get_text_chunks(text)**: Splits the extracted text into manageable chunks.
    ```python
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ```
    Chunk Size (1000 tokens): This size is chosen to balance between providing enough context for the model to understand the text and not exceeding the model's token limit.
Chunk Overlap (100 tokens): This overlap ensures that there is some continuity between chunks, which helps the model maintain context when processing consecutive chunks.
## Model Information

This project uses the [`nlpaueb/legal-bert-base-uncased`](command:_github.copilot.openSymbolFromReferences?%5B%22nlpaueb%2Flegal-bert-base-uncased%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CGIG%5C%5CGIG2%5C%5CChatwithPDFs%5C%5Capp.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FGIG%2FGIG2%2FChatwithPDFs%2Fapp.py%22%2C%22path%22%3A%22%2FC%3A%2FGIG%2FGIG2%2FChatwithPDFs%2Fapp.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A72%2C%22character%22%3A18%7D%7D%5D%5D "Go to definition") model from Hugging Face. The model is a pre-trained BERT model fine-tuned for legal text, which is used to generate answers based on the content of the uploaded PDFs.

Here are some key points why we picked this model:

Domain-Specific Training: The model has been pre-trained on a large corpus of legal documents, making it particularly adept at understanding and processing legal language, terminology, and nuances.

Improved Accuracy: For tasks involving legal text, such as document classification, question answering, and information extraction, this model is likely to perform better than a general-purpose language model due to its specialized training.

Uncased Model: Being an uncased model means it treats uppercase and lowercase letters as the same, which is useful in legal documents where case sensitivity is often not crucial.

Versatility: It can be used for a variety of natural language processing tasks within the legal domain, including but not limited to:

Legal document classification
Named entity recognition (NER) for legal entities
Legal question answering
Summarization of legal documents
Pre-trained: As a pre-trained model, it saves time and computational resources compared to training a model from scratch. Users can fine-tune it further on their specific datasets if needed.

## Dependencies

- streamlit
- PyPDF2
- transformers
- langchain_google_genai
- langchain
- langchain_community
- re
