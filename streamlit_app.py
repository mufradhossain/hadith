import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import chromadb

# Load the model and tokenizer
model_name_or_path = 'intfloat/multilingual-e5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

# Custom embedding function for batch processing
def custom_embedding_function(texts):
    # Tokenize the input sentences
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    # Move tensors to GPU if available
    if torch.cuda.is_available():
        batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
        model.to('cuda')

    # Get model outputs
    with torch.no_grad():
        outputs = model(**batch_dict)

    # Extract the embeddings from the last hidden state (CLS token)
    embeddings = outputs.last_hidden_state[:, 0]

    # Normalize the embeddings
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    # Return embeddings as lists
    return normalized_embeddings.cpu().tolist()

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="hadith-chroma-e5")
collection = client.get_collection(name="hadith")

# Streamlit app
st.title("Hadith Finder")

# User input
query = st.text_input("Enter your query:")

if st.button("Search"):
    # Query the collection
    result = collection.query(
        query_embeddings=custom_embedding_function([query]),
        n_results=5
    )

    # Display results
    for i in range(len(result['metadatas'][0])):
        st.markdown("===" * 50)
        st.write("**Hadith (Bangla):**", result['metadatas'][0][i]['text_bn'])
        st.write("**Hadith (English):**", result['metadatas'][0][i]['text_en'])
        st.write(f"**Hadith No:** {result['metadatas'][0][i]['hadith_no']}")
        st.write(f"**Chapter:** {result['metadatas'][0][i]['chapter']}")
        st.write(f"**Source:** {result['metadatas'][0][i]['source']}")
