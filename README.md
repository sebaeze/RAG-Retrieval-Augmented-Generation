# Retrieval-Augmented Generation (RAG)

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

## How Does It work?

Retrieval:  Fetch data from external knowledge base.
Augmented:  Includes the data previously fetched into the query/prompt.
Generation: The AI Model process the query/prompt and the data from the external knowledge base in order to generate a response.

![rag.png](./diagrams/rag.png)

## Data Ingestions

The process of ingestion consist of separating the data in chunks, creating indexes and insert them into the database. 

Below there is a list of the most popular databases used for RAG:
| Database Name | URL | Description |
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



## What technology is involved?

## Vector Database

[Langchain Vector Data](https://python.langchain.com/v0.1/docs/integrations/vectorstores/)




## References:

- https://huggingface.co/docs/transformers/model_doc/rag
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)