import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bs4
from pinecone import Pinecone
import uuid
from tqdm import tqdm
from groq import Groq
from pinecone import ServerlessSpec


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"Created {len(all_splits)} document chunks.")

#intialize pinecone model
PINECONE_API_KEY = "NA"
GROQ_API_KEY = "NA"
INDEX_NAME = "rag-index"  

pc = Pinecone(api_key=PINECONE_API_KEY)
class TextEmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts)
        return embeddings[0] if len(embeddings) == 1 else embeddings

embedding_model = TextEmbeddingModel()
embedding_dim = 384  

if not pc.has_index(INDEX_NAME):
    print(f"Creating Pinecone index '{INDEX_NAME}' with dimension={embedding_dim}")
    pc.create_index(name=INDEX_NAME, dimension=embedding_dim, metric='cosine',spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1",)
    ,)
else:
    print(f"Index '{INDEX_NAME}' found.")

index = pc.Index(INDEX_NAME)

#upload documents to pinecone

def upload_splits_to_pinecone(splits, index, embedding_model, batch_size=100):
    print(f"Uploading {len(splits)} chunks to Pinecone...")
    for i in tqdm(range(0, len(splits), batch_size)):
        batch = splits[i:i+batch_size]
        vectors = []
        for doc in batch:
            emb = embedding_model.encode(doc.page_content).tolist()
            doc_id = str(uuid.uuid4())
            metadata = {
                'text': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
                'chunk_id': doc_id
            }
            # additional metadata 
            vectors.append({
                'id': doc_id,
                'values': emb,
                'metadata': metadata
            })
        index.upsert(vectors=vectors)
    print("Upload complete.")

upload_splits_to_pinecone(all_splits, index, embedding_model)
#retrieving relevant documents
def retrieve_relevant_documents(query, index, embedding_model, top_k=5):
    query_emb = embedding_model.encode(query).tolist()
    response = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    matches = response.get('matches', [])
    texts = [m['metadata']['text'] for m in matches]
    sources = [m['metadata']['source'] for m in matches]
    print(f"text :{texts}")
    return texts, sources, response
 
def rag_query(user_query, index, embedding_model, groq_client, top_k=5):
    texts, sources, _ = retrieve_relevant_documents(user_query, index, embedding_model, top_k)
    
    context = '\n\n'.join(texts)
    unique_sources = list(set(sources))
    
    sys_prompt = f"""
                    Instructions:
                    - You are a helpful AI assistant who answers based on the context.
                    - Provide detailed and accurate information.
                    - Cite the sources provided.
                    - If information is not in context, say "I don't know".
                    Context:
                    {context}
                    Sources:
                    {', '.join(unique_sources)}
                    """
    
    print(f"Prompt length: {len(sys_prompt)} characters")
    
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1,
        max_tokens=1500
    )
    return {
        'answer': chat_completion.choices[0].message.content
    }

client = Groq(api_key=GROQ_API_KEY)
user_query=st.text_input("Enter the question")
if st.button("Send") and user_query:
    result = rag_query(user_query, index, embedding_model, client)
    st.write(result['answer'])




