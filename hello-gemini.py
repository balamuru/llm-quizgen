import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import easyocr


def interact_with_gemini(prompt_text, api_key, model_name="gemini-1.5-pro"):
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        chain = ChatPromptTemplate.from_template(prompt_text) | llm | StrOutputParser()
        return chain.invoke({})
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_pdf_text(pdf_path):
    documents = PyPDFLoader(pdf_path).load()
    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    text_content = " ".join([doc.page_content for doc in texts])
    return text_content

def process_with_gemini(source_path, get_text, query, api_key, model_name="gemini-1.5-pro"):
    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        question_template = """
        Here is the content of a PDF:
        {file_content}

        Based on this content, answer the following question:
        {question}
        Create output in json format with 4 option with each question json following the format {question_format}.
        Only one option is correct and also provide a hint to the correct answer
        """
        q_fmt="""
            {
                "question": "What battle was considered the turning point of the war?",
                "options": {
                    "a": "The Battle of Bunker Hill",
                    "b": "The Battle of Yorktown",
                    "c": "The Battle of Saratoga",
                    "d": "The Battle of Trenton"
                },
                "answer": "c",
                "hint": "hint to answer"
            }
        """
        text_content = get_text(source_path)
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        prompt_template = ChatPromptTemplate.from_template(template=question_template)
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke({"file_content": text_content, "question": query, "question_format":q_fmt})
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def get_image_text(image_path):
    reader = easyocr.Reader(['en'])
    text_content = " ".join(reader.readtext(image_path, detail=0))
    return text_content


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")

    # prompt = "Tell me a short story about a robot learning to love."
    # result = interact_with_gemini(prompt, api_key)
    # if result:
    #     print(result)

    query = "generate a 10 question multiple-choice quiz about the contents of this document, with the answer keys at the end. also generate a short snippet of the relevant answer context with the each answer in the key"
    file_result = process_with_gemini("/home/vinayb/Downloads/revolution.pdf", get_pdf_text, query, api_key)
    if file_result:
        print(file_result)

    # image_path = "/home/vinayb/Downloads/tea-party.png"
    # image_result = process_image_with_gemini(image_path, query, api_key)
    # if image_result:
    #     print(image_result)
