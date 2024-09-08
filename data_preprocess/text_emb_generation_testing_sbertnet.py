# pip install -U sentence-transformers
# sbert.net documentatin 
# "all-mpnet-base-v2" is a model name : (768',) 768 is the size of the embeddings
# Description:	This model was tuned for semantic search: Given a query/question, 
#               if can find relevant passages. It was trained on a large and diverse set of (question, answer) pairs.
# Max Sequence Length:	512

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

text  = "I am a student  kal I am a student."

embeddings = model.encode(text)
print(embeddings.shape)
print(type(embeddings))
