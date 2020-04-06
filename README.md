## Sentence Transformers serving

see original [paper](https://arxiv.org/abs/1908.10084) 
and [source](https://github.com/UKPLab/sentence-transformers#application-examples)

# Getting Started

## List of Models for Best Sentence Embeddings (taken from [source](https://github.com/UKPLab/sentence-transformers/blob/master/README.md))

**Trained on STS data**

These models were first fine-tuned on the AllNLI datasent, then on train set of STS benchmark. 
They are specifically well suited for semantic textual similarity. For more details, see: 
[sts-models.md](https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md).

- **bert-base-nli-stsb-mean-tokens**: Performance: STSbenchmark: 85.14
- **bert-large-nli-stsb-mean-tokens**: Performance: STSbenchmark: 85.29
- **roberta-base-nli-stsb-mean-tokens**: Performance: STSbenchmark: 85.40
- **roberta-large-nli-stsb-mean-tokens**: Performance: STSbenchmark: 86.31
- **distilbert-base-nli-stsb-mean-tokens**: Performance: STSbenchmark: 84.38


**Performance** 

Extensive evaluation is currently undergoing, but here we provide some preliminary results.

| Model    | STS benchmark | SentEval  |
| ----------------------------------|:-----: |:---:   |
| Avg. GloVe embeddings             | 58.02  | 81.52  |
| BERT-as-a-service avg. embeddings | 46.35  | 84.04  |
| BERT-as-a-service CLS-vector      | 16.50  | 84.66  |
| InferSent - GloVe                 | 68.03  | 85.59  |
| Universal Sentence Encoder        | 74.92  | 85.10  |
|**Sentence Transformer Models**    ||
| bert-base-nli-mean-tokens         | 77.12  | 86.37 |
| bert-large-nli-mean-tokens        | 79.19  | 87.78 |
| bert-base-nli-stsb-mean-tokens    | 85.14  | 86.07 |
| bert-large-nli-stsb-mean-tokens   | 85.29 | 86.66|


## docker-compose parameters
Flask api running on port 5000 will be mapped to outer 5002 port.
(it uses docker-compose version 2.3 which supports `runtime: nvidia` to easily use GPU environment inside container)

Assign name of model that you want to serve to `MODEL` environment variable (default is bert-base-nli-stsb-mean-tokens)
You must remove runtime: nvidia to run docker on cpu. (If it still fails then open Dockerfile and comment section of cuda libraries installation)
```yaml
version: '2.3'
services:
  sentence_transformers_serving:
    container_name: sentence_transformers_serving
    build: .
    ports:
      - "5002:5000"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - MODEL=bert-base-nli-stsb-mean-tokens
```

## Running Docker Container
Probably the easiest way to get started is by using the provided Docker image.
From the project's root directory, the image can be built and running like so:
```
$ docker-compose up --build -d
```
This can take a several minutes to finish. All model files and everything will be downloaded automatically.

## Flask API (inside container)
```python
import os

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from waitress import serve

app = Flask(__name__)


@app.route('/', methods=['POST'])
def encode():
    data = request.json
    sentences = data.get("sentences", [])
    batch_size = int(data.get("batch_size", 8))

    embeddings = encoder.encode(sentences, batch_size=batch_size)
    embeddings = [x.tolist() for x in embeddings]

    return jsonify(embeddings)


if __name__ == '__main__':
    model_name_or_path = os.environ.get('model_name_or_path', "bert-base-nli-stsb-mean-tokens")
    encoder = SentenceTransformer(model_name_or_path=model_name_or_path)

    serve(app, host="0.0.0.0", port=5000)
```

## Curl example to call API

You can simply query from terminal
```bash
curl --location --request POST 'http://0.0.0.0:5002/' \            
    --header 'Content-Type: application/json' \
    --data-raw '{"sentences": ["A man is eating food.", "A man is eating a piece of bread."], "batch_size":8}'
```

## POST request from python
```python
import json
import requests

url = f"http://0.0.0.0:5002/"
data = {"sentences": ["A man is eating food.", "A man is eating a piece of bread."], "batch_size":8}
result = requests.post(url, json=data)

embeddings = json.loads(result.content)
```

## Local Run

## requirements
```text
sentence-transformers==0.2.5.1
Flask==1.1.1
requests==2.23.0
waitress==1.4.3
```

## Sample usage
```python
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
```

# References

https://github.com/UKPLab/sentence-transformers

https://arxiv.org/abs/1908.10084