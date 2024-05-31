CHUNK_SIZE = 1000
OVERLAP_SIZE = 200

template = """
Question: {question}
Context: {context}

You are given a question and a part of pdf as context. You have to answer to the question from the context.
If you are unable to find the answer replay Answser is not present in the given pdf. 
You must not say anything about the context. Don't say words like "Present in Context" "Provided in Context".
You must write the answer in markdown format.
You must write answer with proper formatting,punctuation and grammar.
Use new lines and paragraphs where necessary.
"""
