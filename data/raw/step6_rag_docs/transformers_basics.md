# Transformer Basics

A transformer is a neural network architecture designed for sequence modeling.
In language modeling, a transformer reads tokens and predicts the next token.
The most important building block is self-attention.
Self-attention lets each token compare itself with other tokens in the same sequence and build a weighted summary of relevant context.

In a decoder-only GPT-style model, attention is causal.
That means a token can only attend to earlier tokens, not future ones.
Causal masking is what makes next-token prediction valid during training.
The model also contains feed-forward layers, residual connections, layer normalization, and learned embeddings.

Multi-head attention means the model performs several attention patterns in parallel.
One head may focus on nearby syntax, while another may focus on longer-range dependencies such as subject-verb agreement or topic continuity.
The outputs from multiple heads are combined and passed through later layers.

Transformers scale well because attention and matrix operations work efficiently on modern hardware.
They also avoid the recurrence bottleneck found in older recurrent neural networks.
The tradeoff is that attention cost grows with sequence length, so long contexts can become expensive.

When someone asks what a transformer is, a good short answer is:
"A transformer is a neural network that uses self-attention to model relationships between tokens in a sequence."
