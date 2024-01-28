# link test Images: ../data/data_process_demo/Images

# ----------Libraries----------
import streamlit as st
import numpy as np
from translate import Translator

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import open_clip
from openai import OpenAI
import faiss

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import (ConversationalRetrievalChain, 
                              ConversationChain, 
                              RetrievalQA)

from dotenv import load_dotenv
from os.path import join, dirname
import os
import time

# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
import re

# ----------Local libraries----------
from ProText2text import offline_process_text2text
from ProPic2pic import offline_process_pic2pic, find_similar_image
from ProText2pic import offline_process_text2pic, find_similar_text2pic

st.session_state.theme = "light"


def translate_vietnamese_to_english(text):
    translator= Translator(to_lang="en", from_lang="vi")
    translation = translator.translate(text)
    return translation

# language processing function
def processing_on_1_sent(data):
    # Input: string
    # # lowercase
    # data = data.lower()
    # # remove punctuation and special characters
    # data = re.sub('\W+',' ', data)
    # # remove excess whitespace
    # data = data.strip()
    # # remove StopWord
    # data = ' '.join([word for word in data.split() if word not in stopwords.words("english")])
    # # word tokenize
    # data = word_tokenize(data)
    # data = ' '.join(data)
    # Output: string
    return data

def main():
    # ----------Tiles----------
    header = st.container()
    header.title("IR-based Chatbot System 🤖:speech_balloon:")
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    # st.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    ## Custom CSS for the header
    st.markdown(
        """
        <style>
            div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                position: sticky;
                top: 2.875rem;
                background-color: white;
                z-index: 999;
            }
            .fixed-header {
                border-bottom: 2px solid black;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image('../docs/Userflow IR-based chatbot.png', caption="Interaction with IR-based Chatbot System")
    st.write("""<div class='break'/>""", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            .break {
                border-bottom: 1px solid black;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "client_chatgpt" not in st.session_state:
        st.session_state.client_chatgpt = None

    if "tokenizer_text2pic" not in st.session_state:
        st.session_state.tokenizer_text2pic = None
    if "preprocess_text2pic" not in st.session_state:
        st.session_state.preprocess_text2pic = None
    if "model_text2pic" not in st.session_state:
        st.session_state.model_text2pic = None
    if "db_text2pic" not in st.session_state:
        st.session_state.db_text2pic = None

    if "model_pic2pic" not in st.session_state:
        st.session_state.model_pic2pic = None

    with st.sidebar:
        # ----------Initialization----------
        st.markdown("**Click 'Initialization' button to initialize all settings for the system!**")
        if st.button("Initialization"):
            with st.spinner("Processing"):
                dotenv_path = join(dirname(__file__), '.env')
                load_dotenv(dotenv_path)
                HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

                # embeddings = HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ")
                # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

                # llm = HuggingFaceHub(repo_id = "google/flan-t5-base", model_kwargs = {"temperature": 0.5, "max_length": 64}, huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN) # Tra loi khong hieu qua
                # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 64}, huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN) # Bị giới hạn token
                llm = ChatOpenAI(temperature=0.01, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo").with_config({'callbacks': [ConsoleCallbackHandler()]})

                vectorstore = FAISS.load_local("vectorstores/faiss_text2text", embeddings)
                if st.session_state.db_text2pic == None:
                    st.session_state.db_text2pic = faiss.read_index("vectorstores/index_text2pic_full.bin")
                    

                st.session_state.client_chatgpt = OpenAI(api_key=OPENAI_API_KEY)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                )

                # text2pic
                st.session_state.model_text2pic, _, st.session_state.preprocess_text2pic = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
                st.session_state.tokenizer_text2pic = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
                
                # pic2pic
                st.session_state.model_pic2pic = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

                # Build prompt
                template = """
                    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Always say "thanks for asking!" at the end of the answer. {context}
                    Question: {question}
                    Helpful Answer:
                    """
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

                # Retrieval Question Answering
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                )
                # st.session_state.qa_chain = RetrievalQA.from_chain_type(
                #     llm,
                #     retriever=vectorstore.as_retriever(),
                #     chain_type="map_reduce"
                # )
            st.write("Init completed!")

        # ----------Preprocess data----------
        st.header("Your dataset :open_file_folder:")
        # Documents
        docs = st.file_uploader("Upload your dataset with doc files here and click on 'Process documents', after that, click on 'Initialization':", accept_multiple_files=True, type=["txt", "pdf"])
        if docs and (st.button("Process documents")):
            progress_text = "Operation in progress. Please wait..."
            docs_bar = st.progress(0, text=progress_text)

            offline_process_text2text(docs, docs_bar)

            docs_bar.progress(100, text='Please wait...')
            time.sleep(1)
            st.write("Process documents completed!")
            docs_bar.empty()

        # Images
        imgs_path = st.text_input("Upload path of your dataset with image files here and click on 'Process images', click on 'Initialization':")
        if imgs_path and (st.button("Process images")):

            imgs_bar = st.progress(0, text="Operation in progress. Please wait.")

            offline_process_pic2pic(imgs_path, imgs_bar, st.session_state.model_pic2pic, preprocess_input)

            imgs_bar.progress(80, text="progressing text2pic...")

            offline_process_text2pic(imgs_path, st.session_state.preprocess_text2pic, st.session_state.model_text2pic)
            st.session_state.db_text2pic = faiss.read_index("vectorstores/index_text2pic.bin")

            imgs_bar.progress(100, text="Please wait...")
            time.sleep(1)
            st.write("Process images completed!")
            imgs_bar.empty()
        
        # ----------Upload an image----------
        st.header("Your image :national_park:")
        img_file_buffer = st.file_uploader('Upload an image', type=["png", "jpg"])

    # ----------Display chat messages from history on app rerun----------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == 'user' and 'image' in message:
                st.image(message["image"], width=250)
            if message["role"] == 'assistant' and 'image' in message:
                st.write('Top 1:')
                st.image(message["image"], caption=message["label"], width=250)
                

    prompt = st.chat_input("Talk to IR-based ChatBot")
    prompt_text2pic = None
    if prompt is not None:
        prompt = processing_on_1_sent(prompt)
        prompt_text2pic = translate_vietnamese_to_english(prompt)
        prompt_text2pic = processing_on_1_sent(prompt_text2pic)
    # ----------Chat with ChatGPT----------
    if prompt_text2pic and not img_file_buffer and ('document' not in prompt_text2pic.lower()) and (not any(value in prompt_text2pic.lower() for value in ['look', 'find', 'found']) and not any(value in prompt_text2pic.lower() for value in ['picture', 'image', 'photo', 'img'])):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in st.session_state.client_chatgpt.chat.completions.create(model=st.session_state["openai_model"], 
                                                        messages=[{"role": m["role"], "content": m["content"]}for m in st.session_state.messages], 
                                                        stream=True):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # ----------Chat with Documents----------
    if prompt and not img_file_buffer and ('document' in prompt.lower()):

        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Thinking
            message_thinking = st.empty()
            message_thinking.write("...")

            # llm_response = st.session_state.qa_chain({'query': prompt})
            llm_response = st.session_state.conversation({'question': prompt})

            # Remove the thinking message by re-rendering the UI
            message_thinking.empty()

            # Results
            st.write(llm_response['answer']) # 'result', 'answer'

        st.session_state.messages.append({"role": "assistant", "content": llm_response['answer']})

    # ----------Chat find image from text with Dataset----------
    if prompt_text2pic and not img_file_buffer and (any(value in prompt_text2pic.lower() for value in ['look', 'find', 'found']) and any(value in prompt_text2pic.lower() for value in ['picture', 'image', 'photo', 'img'])):
        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        similarities = find_similar_text2pic(prompt_text2pic, st.session_state.db_text2pic, st.session_state.model_text2pic, st.session_state.tokenizer_text2pic) #.split(":")[1]

        with st.chat_message("assistant"):
            st.write("Here are the top 5 images in your dataset similar to the caption:")
            for i, (filepath, similarity) in enumerate(similarities):
                if i == 0:
                    st.session_state.messages.append({"role": "assistant", 
                                        "content": "Here are the top image in your dataset similar to the caption:", 
                                        "content_1":"Top 1:", 
                                        'image': filepath,
                                        'label': filepath.split('/')[3] + ' (similarity: ' + str(round(similarity*100, 3)) +'%)'})
                st.write(f"Top {i+1}:")
                st.image(filepath, filepath.split('/')[3] + ' (similarity: ' + str(round(similarity*100, 3)) +'%)', width=250)

    # ----------Chat with Images----------
    if prompt and img_file_buffer:
        with st.chat_message("user"):
            st.write(prompt)
            st.image(img_file_buffer, width=250)

        st.session_state.messages.append({"role": "user", "content": prompt, 'image': img_file_buffer})

        
        with st.chat_message("assistant"):
            # Thinking
            message_thinking = st.empty()
            message_thinking.write("...")

            query_image = cv2.imread('../data/data_imgs_test/' + img_file_buffer.name)
            query_image = cv2.resize(query_image, (224, 224))
            query_image = preprocess_input(query_image)
            query_feature = st.session_state.model_pic2pic.predict(np.expand_dims(query_image, axis=0)).flatten()
            similarities = find_similar_image(query_feature, top_n=5)

            # Remove the thinking message by re-rendering the UI
            message_thinking.empty()

            # Results
            st.write("Here are the top 5 images in your dataset similar to the image you sent:")
            for i, (filepath, label, similarity) in enumerate(similarities):
                if i == 0:
                    st.session_state.messages.append({"role": "assistant", 
                                        "content": "Here are the top image in your dataset similar to the image you sent:", 
                                        "content_1":"Top 1:", 
                                        'image': filepath,
                                        'label': label + ' (similarity: ' + str(round(similarity*100, 3)) +'%)'})
                    
                st.write(f"Top {i+1}:")
                st.image(filepath, caption=label + ' (similarity: ' + str(round(similarity*100, 3)) +'%)', width=250)

if __name__ == '__main__':
    main()
