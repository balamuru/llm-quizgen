# python-ai

An LLM (Gemini) powered Quiz generator I wrote to help my child with revising schoolwork topics

## Features
* Supports the following file modes
  * RAG Mode - attach desired File (text, pdf, jpg/png image)
  * LLM Mode just enter the topic and any other details in the text prompt
* Targeted topics - specify the specific topic within the attached document to generate a quiz
  * This is optional in RAG mode and mandatory in LLM mode

## Tech Stack
* language - Python 
* llm framework - Langchain / Gemini / GenAI
* web framework - Streamlit

## Local Dev Setup
* Setup and activate python virtual environment and install dependencies
```
git clone https://github.com/balamuru/llm-quizgen.git
cd llm-quizgen
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
* Run `streamlit run gemini-quizgen.py`
* On the UI Specify `GEMINI_API_KEY`  - Get this from Google AI Studio

## TODO
* Summarizer
* Video transcript input for quiz generator