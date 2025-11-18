import os
import pickle
import time

from dotenv import load_dotenv

from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. Load environment variables (OPENAI_API_KEY from .env)
load_dotenv()

# 2. Ask user for URLs
print("=== RockyBot: News Research Tool (CLI Version) ===")
print("Enter up to 3 news article URLs (press Enter to skip a slot):")

urls = []
for i in range(3):
    url = input(f"URL {i+1}: ").strip()
    if url:
        urls.append(url)

if not urls:
    print("No URLs provided. Exiting.")
    exit(0)

file_path = "faiss_store_openai.pkl"

# 3. Initialize LLM
llm = OpenAI(temperature=0.9, max_tokens=500)
             

# 4. Load and process URLs
print("\n Loading data from URLs...")
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print("    ✅ Data loaded.")

# 5. Split text into chunks
print(" Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000
)
docs = text_splitter.split_documents(data)
print(f"    ✅ Created {len(docs)} chunks.")

# 6. Create embeddings and FAISS index
print("Creating embeddings and building FAISS vector store...")
embeddings = OpenAIEmbeddings()
vectorstore_openai = FAISS.from_documents(docs, embeddings)
print("    ✅ FAISS index built.")

# 7. Save FAISS index to disk
print("Saving FAISS index to file...")
with open(file_path, "wb") as f:
    pickle.dump(vectorstore_openai, f)
print(f"    ✅ Saved FAISS index to '{file_path}'.\n")

time.sleep(1)

# 8. Question-answer loop
print("=== You can now ask questions about the articles ===")
print("Type 'exit' or 'quit' to end.\n")

# Load the vectorstore from file (just to mimic the original behavior)
with open(file_path, "rb") as f:
    vectorstore = pickle.load(f)

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=retriever
)

while True:
    query = input("Question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Goodbye! 👋")
        break

    if not query:
        continue

    print("\nThinking...\n")
    result = qa_chain({"question": query}, return_only_outputs=True)

    # result format: {"answer": "...", "sources": "..."}
    answer = result.get("answer", "")
    sources = result.get("sources", "")

    print("=== Answer ===")
    print(answer)
    print()

    if sources:
        print("=== Sources ===")
        # Sources may be separated by newlines
        sources_list = sources.split("\n")
        for src in sources_list:
            if src.strip():
                print("-", src.strip())
        print()



