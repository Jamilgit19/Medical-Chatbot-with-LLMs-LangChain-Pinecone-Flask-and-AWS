from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from src.prompt import prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize embeddings model
embeddings = download_embeddings()

# Connect to existing Pinecone index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Build RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User input: {msg}")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print(f"Response: {answer}")
        return str(answer)
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            return "⏳ The API rate limit has been reached. Please wait a moment and try again."
        return "Sorry, something went wrong while processing your question. Please try again."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
