retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "What is the main topic of the book?"
docs = retriever.invoke(query)

for d in docs:
    print(d.metadata, d.page_content[:300])
