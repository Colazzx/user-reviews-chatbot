import streamlit as st
from bot_qna import handle_query, get_qa_chain

# Initialize session state variables for conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = get_qa_chain()

# Function to handle clearing the input field after submission
def submit_question():
    user_input = st.session_state.get('user_input', '')
    expected_answer = st.session_state.get('expected_answer', None)

    if user_input:
        # Get the response from the chatbot
        answer, score = handle_query(user_input, st.session_state['qa_chain'], expected_answer)

        # Append the interaction to the conversation history
        st.session_state['conversation'].insert(0, {
            "user": user_input,
            "bot": answer,
            "score": score if expected_answer else None
        })

        # Set a flag to reset the input
        st.session_state['input_submitted'] = True

# Function to clear chat history
def clear_chat_history():
    st.session_state['conversation'] = []

# Streamlit page layout
st.set_page_config(page_title="Spotify Q&A Chatbot", page_icon="🎧", layout="centered")

# Title and description
st.title("Spotify Q&A Chatbot 🎧")
st.write("Ask me anything about Spotify user reviews!")

# Input field for user's question and expected answer (optional)
user_input = st.text_input("You: ", key="user_input", placeholder="Ask your question here...")
expected_answer = st.text_input("Expected Answer (optional, for quality scoring): ", key="expected_answer")

# Submit and Clear buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Submit"):
        submit_question()
with col2:
    if st.button("Clear Chat History"):
        clear_chat_history()

# Display conversation history in a chat format
st.write("### Chat History")
for interaction in st.session_state['conversation']:
    st.write(f"**You:** {interaction['user']}")
    st.write(f"**Bot:** {interaction['bot']}")

# Check if an expected answer was provided for quality scoring
if st.session_state['conversation']:
    last_interaction = st.session_state['conversation'][0]
    score = last_interaction.get('score', None)

    if score:
        st.write("### Quality Scoring")
        st.write(f"Relevance Score: {score['relevance_score']:.2f}")
        st.write(f"Accuracy Score: {score['accuracy_score']:.2f}")
        st.write(f"Clarity Score: {score['clarity_score']:.2f}")
        st.write(f"Final Score: {score['final_score']:.2f}")
    else:
        st.write("No quality score available for this interaction.")
