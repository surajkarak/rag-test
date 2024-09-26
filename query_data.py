import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
#from langchain_community.llms.ollama import Ollama
from transformers import AutoModelForCausalLM, AutoTokenizer

from get_embedding_function import SentenceTransformerEmbeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = SentenceTransformerEmbeddings()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Load the model and tokenizer
    model_name = "gpt2"  # Replace with your desired model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)


    # Encode the prompt into tokens
    inputs = tokenizer(prompt, return_tensors="pt")


    # Generate response using the model
    outputs = model.generate(**inputs, max_length=1024, do_sample=True)

    
    # Decode the generated tokens into text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response


def main():
    st.title("RAG-based Query Answering")

    # Text input for the query
    query_text = st.text_input("Enter your question:")

    # Button to trigger the query
    if st.button("Submit"):
        if query_text:
            # Call the query_rag function and display the result
            with st.spinner("Searching and generating response..."):
                response = query_rag(query_text)
                st.write(response)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
