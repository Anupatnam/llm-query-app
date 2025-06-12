from flask import Flask, request, render_template
import requests
import google.generativeai as genai
from utils.logger import setup_logger
from utils.helpers import get_env_var
from dotenv import load_dotenv
from utils.faiss_index import semantic_search

load_dotenv()
logger = setup_logger()

app = Flask(__name__)
genai.configure(api_key=get_env_var("GEMINI_API_KEY"))
GROQ_API_KEY = get_env_var("GROQ_API_KEY")

def query_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "Gemini failed to respond."

def query_groq(prompt: str) -> str:
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq Error: {e}")
        return "Groq failed to respond."

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['user_query']
    logger.info(f"User Query: {user_query}")

    context = semantic_search(user_query)
    logger.info(f"Semantic Context: {context}")

    full_prompt = f"{context}\n\nUser: {user_query}"

    groq_res = query_groq(full_prompt)
    gemini_res = query_gemini(full_prompt)

    logger.info(f"Groq Response: {groq_res}")
    logger.info(f"Gemini Response: {gemini_res}")

    return render_template("index.html", groq_response=groq_res, gemini_response=gemini_res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
