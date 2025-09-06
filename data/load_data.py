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
def tokenize(text):
    return text.split()


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

