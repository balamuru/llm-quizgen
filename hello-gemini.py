import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def interact_with_gemini(prompt_text, api_key):
    try:
        # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key) # 15 RPM, multimodal
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key) # 2 RPM, multimodal
        langchain_prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = langchain_prompt | llm | StrOutputParser()
        response = chain.invoke({})
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None





def process_pdf_with_gemini(pdf_path, query, model_name="gemini-pro"):
    """
    Processes a PDF file using Gemini via LangChain, answering a query based on the PDF's content.

    Args:
        pdf_path (str): The path to the PDF file.
        query (str): The query related to the PDF content.
        model_name (str): The Gemini model to use (default: "gemini-pro").

    Returns:
        str: The response from Gemini, or None if an error occurred.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")

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

if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")

    prompt = "Tell me a short story about a robot learning to love."
    result = interact_with_gemini(prompt, api_key)
    if result:
        print(result)
