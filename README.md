## About

This is a simple step-by-step example of Retrieval-Augmented Generation (RAG) created by [ykimura517](https://twitter.com/yk_llm_gpt).

For more details about this repository, please visit [my blog post](https://zenn.dev/articles/8af7cbf526c2e1).

If you have any questions, feel free to reach out to [me](https://twitter.com/yk_llm_gpt).

## Usage

### Set Your OpenAI API Key

```
export OPENAI_API_KEY={{your-api-key}}
```


### Install Required Libraries

```
pip install -r requirements.txt
```

### Save Sample Data

```
python3 embedder.py
```

By running this script, `sample_data.json` will be saved in the local directory. It contains sample texts along with their vector data.


### Run Sample Code

```
python3 main.py
```

The detailed steps in this script are as follows:  

1. Embed the user query using OpenAI Embeddings.
2. Search for the nearest data in sample_data.json based on cosine similarity.
3. Use the data retrieved in step 2 to construct a prompt and generate an answer with GPT.
  
That's all! Happy hacking!