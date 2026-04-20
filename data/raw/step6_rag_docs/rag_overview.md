# Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, combines a retriever with a generator.
Instead of asking the language model to answer from parametric memory alone, the system first searches a document collection for relevant passages.
Those passages are then inserted into the prompt as context for generation.

A simple RAG pipeline usually has four steps:
1. ingest documents
2. split documents into chunks
3. retrieve the most relevant chunks for a query
4. generate an answer conditioned on the retrieved context

Chunking matters because retrieval usually works better on passage-sized text than on full documents.
If chunks are too short, important context may be split apart.
If chunks are too long, retrieval becomes noisy and prompt space is wasted.
Chunk overlap is often added so details near boundaries are less likely to be lost.

The retriever does not have to be fancy.
A lexical baseline such as BM25 is often the best first implementation because it is fast, understandable, and easy to debug.
Dense embedding retrieval can be added later if recall becomes a problem.

RAG is most useful when the answer should be grounded in an external document set.
It improves transparency because the system can show which chunks were retrieved.
It also makes updates easier because changing the documents can update behavior without retraining the model.

RAG is not a guarantee of correctness.
If retrieval fails, generation may still be wrong.
That is why a grounded prompt should tell the model to say "I do not know" when the provided context is insufficient.
