# Retrieval-Augmented Generation (RAG)

- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Purpose](#purpose)
  - [What is RAG?](#what-is-rag)
  - [Why to use RAG?](#why-to-use-rag)
  - [What RAG means?](#what-rag-means)
  - [How Does It work?](#how-does-it-work)
    - [Data Ingestions](#data-ingestions)
    - [Retrieval process](#retrieval-process)
      - [Criteria for ranking data](#criteria-for-ranking-data)
    - [Generation](#generation)
  - [Most popular Vector Databases](#most-popular-vector-databases)
  - [References:](#references)
  

## Purpose

The purpose of this repo is to guide people who are getting into the world of RAG technology and AI.

## What is RAG?

RAG is a technique that integrates the AI Model with external data in order to enhance the result without the need to re-train the model.
The process consist of retrieving data from an external store, such as a vector database, to include in the prompt or context to be processed by the AI Model.

## Why to use RAG?

RAG technology in intended to solve limitations of usin AI Models (Small and Large Language Models), such as:

- Pre-trained or static data: AI Model are trained with static data which became outdated.
- Size of AI Model: the use of RAG allows the selection of small size Language Model due to the capability of providing up-to-date data.
- Domain-Specific customization: RAG technology is usefull for retrieving data about specific domains without need to re-train a model.
- Combines context data with up-to-date information.

## What RAG means?

Retrieval:  Fetch data from external knowledge base.
Augmented:  Includes the data previously fetched into the query/prompt.
Generation: The AI Model process the query/prompt and the data from the external knowledge base in order to generate a response.

![rag.png](./diagrams/rag.png)

## How Does It work?

### Data Ingestions

The process of ingestion consist of:
- Collect the data from the documents, websites, and so on.
- Clean the information by remmoving unnecesary data.
- Separate the data into chunks. 
- Embedding: process the data by an embedding model, which converts It into vector representation of the data, capturing the semantic meaning.
- The embedded data is loaded into a database, usually a Vector Database.
- Index the embedding data.


### Retrieval process

The retrieval phase is the process of searching and identifyig relevant information in the database. 

The steps executed by the retrieval are:
- Query formulation: Interpret and understand the input query from the user in order to use It as input for generating a query to a database and/or external knowledge database.
- Retrieval: The query is used for retrival of information.
- Ranking: The information is ranked based on the query.
- Selection: The top-ranked information is selected.
- Context generation: Within the selected information, the process generate context using the most relevant data, which is included in the input to the AI Model.
  
#### Criteria for ranking data

- Relevance to the query
- Context of the conversation
- Up-to-date data. In some cases, the process might prioritize how recent is the data
- Reliability of the data. The system might consider the reliability of the source
- Conversation history

### Generation 

The generation step consist of sending the contextual data created in the retrieval process in combination with the user's input message to the AI Model, which generates the final response.

## Most popular Vector Databases

Below there is a list of the most popular databases used for RAG:
| Database Name | <div style="width: 50px">URL</div> | <div style="width: 150px">Description</div> |
|---|---|---|
| Pinecone | [https://www.pinecone.io/](https://www.pinecone.io/) | A fully managed vector database designed for high-performance similarity search, ideal for real-time applications. |
| Weaviate | [https://weaviate.io/](https://weaviate.io/) | A cloud-native, real-time vector search engine with GraphQL and RESTful APIs, supporting hybrid search. |
| ChromaDB | [https://www.trychroma.com/](https://www.trychroma.com/) | An open-source embedding database that's easy to use and deploy, suitable for development and smaller applications. |
| FAISS (Facebook AI Similarity Search) | [https://faiss.ai/](https://faiss.ai/) | A library for efficient similarity search and clustering of dense vectors, highly optimized for performance. |
| Milvus | [https://milvus.io/](https://milvus.io/) | An open-source vector database built for large-scale vector similarity search, supporting various indexing and query types. |
| Qdrant | [https://qdrant.tech/](https://qdrant.tech/) | A vector similarity search engine and database, providing a scalable and production-ready solution for vector search. |
| Vespa | [https://vespa.ai/](https://vespa.ai/) | An open-source big data serving engine that also excels at vector search, supporting hybrid search and complex queries. |
| MongoDB Atlas Vector Search | [https://www.mongodb.com/products/platform/atlas-vector-search](https://www.mongodb.com/products/platform/atlas-vector-search) | Integrates vector search capabilities within MongoDB Atlas, allowing for unified data management and querying. |
| Supabase vector | [https://supabase.com/docs/guides/database/extensions/vectors](https://supabase.com/docs/guides/database/extensions/vectors) | Postgres extensions that enable vector storage and searching, integrated directly into the Supabase platform. |
| Redis Vector | [https://redis.com/docs/stack/search/reference/vectors/](https://redis.com/docs/stack/search/reference/vectors/) | Redis extends its data structures to include vectors, allowing for fast vector similarity searches alongside traditional Redis operations. |


## References:

- https://huggingface.co/docs/transformers/model_doc/rag
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)