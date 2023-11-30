import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Enterprise AI Solutions by Avinash Polineni", page_icon=":briefcase:")
st.title("Enterprise AI Solutions by Avinash Polineni")

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Load data from a more focused CSV file
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='enterpriseData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Keywords', 'Category']
})

# Load and process data
data = loader.load()
db = FAISS.from_documents(data, embeddings)

# UI for user input
st.header("Explore AI-Driven Insights Across Enterprise Domains")
st.markdown("""
### How It Works:
Enter a query related to enterprise topics like 'Supply Chain Optimization', 'Cybersecurity', or 'Market Analysis'. The AI will provide similar keywords and categories, offering insights and expanding your understanding of the topic.

### Example Queries:
- "Cybersecurity in Supply Chains"
- "AI in Customer Relationship Management"
- "Data Analytics in Market Forecasting"

**Now, try entering your own query!**
""")

user_input = st.text_input("Your Query: ", key='user_input')
submit = st.button('Generate Insights')

if submit:
    # Find similar entries
    similar_entries = db.similarity_search(user_input)
    if similar_entries:
        st.subheader("AI Generated Insights")
        for entry in similar_entries[:3]:  # Show top 3 matches
            st.write(entry.page_content)
    else:
        st.write("No relevant matches found.")

# Feedback mechanism
st.write("Was this information helpful?")
feedback = st.radio("", ('Yes', 'No'), key='feedback')
if feedback:
    st.write("Thank you for your feedback!")

# Footer and additional resources
st.sidebar.markdown("### Additional Resources")
st.sidebar.markdown("[Checkout Avinash's Hugging Face Profile](https://huggingface.co/AvinashPolineni)")
st.sidebar.markdown("[Checkout Avinash's GitHub Profile](https://github.com/polineniavinash)")
st.sidebar.markdown("[Contact Me on LinkedIn](https://linkedin.com/in/avinash-polineni/)")
st.markdown("---")
st.caption("© 2023 Avinash Polineni.")

# Reminder: Update 'enterpriseData.csv' with more relevant entries specific to blockchain and supply chain optimization.
