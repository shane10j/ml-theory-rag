import nltk


#Downloading the Guteberg corpus for use in the initial, barebones pipeline
#Only 18 stories
def load_gutenberg_data():
    nltk.download('gutenberg')
    from nltk.corpus import gutenberg

    files = gutenberg.fileids()
    texts = [gutenberg.raw(fileid) for fileid in files]
    return files, texts


#Tokenize
def tokenize_helper(text):
    return text.split()

def tokenize(texts):
    tokenized = []
    for text in texts:
        tokenized += tokenize_helper(text)
    return tokenized

#Chunking the tokensj
def chunk(tokens, size, overlap): #Overlap to deal with boundary problem
    chunks = []
    stride = size - overlap
    for i in range(0, len(tokens)-size, stride):
        if i+size > len(tokens):
            chunks.append(tokens[i:])
        else:
            chunks.append(tokens[i:i+size])   
    return chunks

