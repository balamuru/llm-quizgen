import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from PIL import Image
import pytesseract

def interact_with_gemini(prompt_text, api_key, model_name="gemini-1.5-pro"):
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        langchain_prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = langchain_prompt | llm | StrOutputParser()
        response = chain.invoke({})
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def process_pdf_with_gemini(pdf_path, query, api_key, model_name="gemini-1.5-pro"):
    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        text_content = " ".join([doc.page_content for doc in texts])

        prompt_template = ChatPromptTemplate.from_template(
            """
            Here is the content of a PDF:
            {pdf_content}

            Based on this content, answer the following question:
            {question}
            """
        )

        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"pdf_content": text_content, "question": query})

        return response

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None


def process_image_with_gemini(image_path, query, api_key, model_name="gemini-1.5-pro"):
    """
    Processes an image file using Gemini via LangChain, answering a query based on the image's text content.

    Args:
        image_path (str): The path to the image file.
        query (str): The query related to the image content.
        api_key (str): The API key for Gemini.
        model_name (str): The Gemini model to use (default: "gemini-pro").

    Returns:
        str: The response from Gemini, or None if an error occurred.
    """
    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Extract text from image using pytesseract
        image = Image.open(image_path)
        text_content = pytesseract.image_to_string(image)

        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

        prompt_template = ChatPromptTemplate.from_template(
            """
            Here is the content of an image:
            {image_content}

            Based on this content, answer the following question:
            {question}
            """
        )

        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"image_content": text_content, "question": query})

        return response

    except Exception as e:
        print(f"Error processing image: {e}")
        return None



if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")

    # prompt = "Tell me a short story about a robot learning to love."
    # result = interact_with_gemini(prompt, api_key)
    # if result:
    #     print(result)

    pdf_path = "/home/vinayb/Downloads/revolution.pdf"
    # pdf_path = "/home/vinayb/Downloads/tea-party.png"
    # query = "What is the main topic of the PDF?"
    query = "generate a multiple-choice quiz about the contents of this document"
    pdf_result = process_pdf_with_gemini(pdf_path, query, api_key)
    if pdf_result:
        print(pdf_result)