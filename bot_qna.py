from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAI
from textblob import TextBlob
from secret_key import openai_key
import faiss
from tqdm import tqdm
import torch
import os
import pandas as pd

# API Key for OpenAI
os.environ['OPENAI_API_KEY'] = openai_key

# Set up the OpenAI chat model
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.2)
# llm = OpenAI(temperature=0.3, model="gpt-4o", max_tokens=1000)

# Setting up for the Faiss Vector Database
embeddings = "sentence-transformers/all-MiniLM-L6-v2"
vectordb_file_path = "faiss_index"

# Initialize the embeddings model
embedding_model = HuggingFaceEmbeddings(model_name=embeddings)

# Initialize QA Chain
def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embedding_model, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever()

    # Define the prompt template
    prompt_template = """
    You are an assistant designed to answer questions based on user reviews of a music streaming application.

    **Layer 1: Contextual Understanding**
    Please read the following user reviews carefully and provide precise answers to the questions based on the source document context (Spotify User Reviews Database) provided. Do not answer any questions outside this context.

    **Layer 2: User Instruction**
    Respond in a friendly and professional tone. If the answer involves comparisons to other music streaming platforms, provide specific examples from the source document context. 

    **Layer 3: Response Guidelines**
    If the answer is not explicitly found in the source document context, kindly state: "I'm sorry, I don't have that information." Please do not fabricate any answers or discuss unrelated topics.

    CONTEXT: {context}

    QUESTION: {question}
    """

    # Create the PromptTemplate
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Set up the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

# Initialize the memory for conversation
memory = ConversationBufferWindowMemory(k=5)  # Keep the last 5 interactions

# Function to analyze sentiment
def analyze_sentiment(review_text):
    analysis = TextBlob(review_text)
    sentiment_score = analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

    # Create a more informative response based on the sentiment score
    if sentiment_score > 0.2:
        sentiment_description = "positive"
    elif sentiment_score < -0.2:
        sentiment_description = "negative"
    else:
        sentiment_description = "neutral"

    return f"Based on the review, the sentiment is {sentiment_description} with a score of {sentiment_score:.2f}."

# Function to calculate similarity between two text strings (using embedding model)
def calculate_similarity_score(answer, expected_answer, embedding_model):
    # Skip if expected_answer is None
    if expected_answer is None:
        return 0

    # Proceed with embedding and similarity score calculation
    answer_embedding = embedding_model.embed_query(answer)
    expected_answer_embedding = embedding_model.embed_query(expected_answer)

    return cosine_similarity([answer_embedding], [expected_answer_embedding])[0][0]

# Function to score the response quality
def quality_scoring(response, expected_answer, embedding_model):
    # If expected_answer is None, skip similarity score calculation
    if expected_answer is None:
        relevance_score = 0  # Default relevance score when no expected answer is provided
    else:
        relevance_score = calculate_similarity_score(response, expected_answer, embedding_model)

    # Accuracy and clarity scoring can proceed as usual
    accuracy_score = 1 if "factually correct" in response else 0.8
    clarity_score = 1 if len(response) > 50 else 0.7

    # Calculate the final score based on weighted components
    final_score = (0.5 * relevance_score) + (0.3 * accuracy_score) + (0.2 * clarity_score)

    return {
        "relevance_score": relevance_score,
        "accuracy_score": accuracy_score,
        "clarity_score": clarity_score,
        "final_score": final_score
    }

# Function to summarize reviews
def summarize_reviews(qa_chain):
    summary_query = "Can you summarize the key points from the reviews?"
    summary_response = qa_chain({"query": summary_query})
    memory.save_context({"input": summary_query}, {"output": summary_response['result']})
    return summary_response['result']

# Function to analyze trends
def analyze_trends(qa_chain):
    trends_query = "What trends are users discussing in their reviews?"
    response = qa_chain({"query": trends_query})
    memory.save_context({"input": trends_query}, {"output": response['result']})
    return response['result']

# Function to handle the question and update memory
def handle_query(query, qa_chain, expected_answer=None):
    if "summarize" in query.lower():
        answer = summarize_reviews(qa_chain)
        score = None  # No score for summaries
    elif "trends" in query.lower() or "patterns" in query.lower():
        answer = analyze_trends(qa_chain)
        score = None  # No score for trends
    else:
        # Process the query through the chain
        response = qa_chain({"query": query})

        # Get the actual answer from the response
        answer = response.get('result', "I'm sorry, I don't have that information.")

        # Trim the expected answer if it's too long (let's say > 300 characters)
        if expected_answer and len(expected_answer) > 300:
            expected_answer = expected_answer[:300] + "..."  # Trimming the long expected answer

        # Quality scoring of the answer, if expected_answer is provided and not too long
        if expected_answer:
            try:
                score = quality_scoring(answer, expected_answer, embedding_model)
            except Exception as e:
                print(f"Error in quality scoring: {e}")
                score = None
        else:
            score = None

        # Save the context in memory after getting the response
        memory.save_context({"input": query}, {"output": answer})

    return answer, score

# Example usage of the chain
if __name__ == "__main__":
    qa_chain = get_qa_chain()
    query = "What are the primary reasons users express dissatisfaction with the service?"
    expected_answer = "Users are dissatisfied due to poor music recommendations and frequent ads."  
    answer, score = handle_query(query, qa_chain, expected_answer)
    print(f"Answer: {answer}")
    print(f"Score: {score}")

