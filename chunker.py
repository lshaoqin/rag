def chunk_by_words(text, chunk_size, overlap_size=0):
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

