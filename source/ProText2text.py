from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
from os.path import join, dirname
import os

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_txt_text(txt_docs):
    text = ""
    for doc in txt_docs:
        if doc.type == 'text/plain':
            txt = doc.read().decode('utf-8')
            # Process the contents of the file
            text += txt +'\n'
        else:
            pass
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def offline_process_text2text(docs, docs_bar):
    file_extension = os.path.splitext(docs[0].name)[1]
    print(file_extension)

    docs_bar.progress(20, text='Getting raw text.')
    if file_extension == ".txt":
        # get txt text
        raw_text = get_txt_text(docs)
    elif file_extension == ".pdf":
        # get pdf text
        raw_text = get_pdf_text(docs)
    else:
        return False  # Unsupported file format
    
    if raw_text:
        # get the text chunks
        docs_bar.progress(40, text='Splitting chunks.')
        text_chunks = get_text_chunks(raw_text)

        if text_chunks:
            # create vector store
            docs_bar.progress(60, text='Saving chunks to vectorstore.')
            vectorstore = get_vectorstore(text_chunks)

            if vectorstore:
                vectorstore.save_local("vectorstores/faiss_text2text")
                return True  # Processing completed successfully
            else:
                return False  # Failed to create vector store
        else:
            return False  # Failed to get text chunks
    else:
        return False  # Failed to get raw text
    