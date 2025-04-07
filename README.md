# python-ai

Just an LLM (Gemini) powered Quiz generator I wrote to help my child with revising schoolwork topics

## Features

* Supports the following file modes
  * RAG Mode - attach desired File (text, pdf, jpg/png image)
  * LLM Mode just enter the topic and any other details in the text prompt
* Targeted topics - specify the specific topic within the attached document to generate a quiz
  * This is optional in RAG mode and mandatory in LLM mode

## Tech

* Python
* Langchain
* Gemini / GenAI
* Streamlit to power the stateful webapp

## Setup
### Local Dev

* Setup and activate python virtual environment and install dependencies
```
git clone https://github.com/balamuru/PythonAI.git
cd PythonAI
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

* Define environment variables (in IDE or OS env)
  * `GEMINI_API_KEY`  - Get this from Google AI Studio

* Run `streamlit run gemini-quizgen.py`

## TODO
* Summarizer
* Organize quiz by topics
* Targeted topics within document
* Video summarizer / quiz