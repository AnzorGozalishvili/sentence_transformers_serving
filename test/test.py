from sentence_transformers import SentenceTransformer

"""
    Available Model Names:
        "bert-base-nli-stsb-mean-tokens"
        "bert-large-nli-stsb-mean-tokens"
        "roberta-base-nli-stsb-mean-tokens"
        "roberta-large-nli-stsb-mean-tokens"
        "distilbert-base-nli-stsb-mean-tokens"
"""

model = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.']

corpus_embeddings = model.encode(corpus)

print(corpus_embeddings)
