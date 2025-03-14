# Retrieval-Augmented Generation (RAG)

This repo is intended to teach about RAG technology

## What is RAG?

RAG is a technique that integrates the AI Model with external data in order to enhance the result without the need to re-train the model.
The process consist of retrieving data from an external store, such as a vector database, to include in the prompt or context to be processed by the AI Model.


## How Does It work?

Retrieval:  Fetch data from external knowledge base.
Augmented:  Includes the data previously fetched into the query/prompt.
Generation: The AI Model process the query/prompt and the data from the external knowledge base in order to generate a response.

![rag.png](./diagrams/rag.png)

## Why to use RAG?

RAG technology in intended to solve limitations of usin AI Models (Small and Large Language Models), such as:

- Pre-trained or static data: AI Model are trained with static data which became outdated.
- Size of AI Model: the use of RAG allows the selection of small size Language Model due to the capability of providing up-to-date data.
- Domain-Specific customization: RAG technology is usefull for retrieving data about specific domains without need to re-train a model.

## What technology is involved?

## Vector Database

[Langchain Vector Data](https://python.langchain.com/v0.1/docs/integrations/vectorstores/)




## References:

- https://huggingface.co/docs/transformers/model_doc/rag
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)