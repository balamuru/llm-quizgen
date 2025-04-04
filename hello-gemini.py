import streamlit as st
import easyocr
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def get_pdf_text(pdf_path):
    documents = PyPDFLoader(pdf_path).load()
    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    text_content = " ".join([doc.page_content for doc in texts])
    return text_content

def get_image_text(image_path):
    reader = easyocr.Reader(['en'])
    text_content = " ".join(reader.readtext(image_path, detail=0))
    return text_content

def get_file_text(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
        return text

def process_with_gemini(source_path, get_text, query, api_key, output_type="json", model_name="gemini-1.5-pro"):
    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        question_template = """
        Here is the content of a PDF:
        {file_content}

        Based on this content, answer the following question:
        {question}
        Create a JSON object  (not markdown) ready to be consumed by a webclient with each question json following the format {question_format}.
        Only one option is correct
        Provide a reason to justify why the answer is correct
        Also provide a hint to the correct answer
        # """
        q_fmt = """
            `{
                "question": "What battle was considered the turning point of the war?",
                "options": {
                    "a": "The Battle of Bunker Hill",
                    "b": "The Battle of Yorktown",
                    "c": "The Battle of Saratoga",
                    "d": "The Battle of Trenton"
                },
                "answer": "c"
                "reason": "because it was important"
                "hint": "hint to answer"
            }`
        """

        text_content = get_text(source_path)
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        prompt_template = ChatPromptTemplate.from_template(template=question_template)

        if output_type == "json":
            parser = JsonOutputParser()
        elif output_type == "text":
            parser = StrOutputParser()
        else:
            raise TypeError(f"unknown output type {output_type}")

        chain = prompt_template | llm | parser
        return chain.invoke({"file_content": text_content, "question": query, "question_format": q_fmt})
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def main():
    st.title("Quiz Generator with Gemini")
    st.write("Upload a file and generate a quiz based on its content.")

    api_key = st.text_input("Enter your GEMINI_API_KEY", type="password")
    query = st.text_input("Enter your query", "generate a multiple-choice quiz about the contents of this document")
    output_type = "json"
    #output_type = st.selectbox("Select output type", ["json", "text"], index=0) # text is not applicable for this app
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png", "txt"])

    if uploaded_file is not None and api_key:
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # very simplistic file type detection logic but should work ok for now
        if uploaded_file.name.lower().endswith('.pdf'):
            result = process_with_gemini(file_path, get_pdf_text, query, api_key, output_type)
        elif uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            result = process_with_gemini(file_path, get_image_text, query, api_key, output_type)
        elif uploaded_file.name.lower().endswith('.txt'):
            result = process_with_gemini(file_path, get_file_text, query, api_key, output_type)
        else:
            st.error("Unsupported file type")
            return

        if result:
            if output_type == "json":
                st.json(result)
            else:
                st.text(result)
        else:
            st.error("Error processing file")

if __name__ == "__main__":
    main()