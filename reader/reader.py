def construct_prompt(query, retrieved_chunks):
    prompt = "Use the following chunks to answer the question: " + query + "\n"
    for i in range(len(retrieved_chunks)):
        chunk = " ".join(retrieved_chunks[i])
        prompt += chunk + "\n"
    return prompt