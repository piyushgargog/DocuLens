# Retrieval Evaluation

Document: `sample_docs/dev_real_world_document.pdf` — 39 labelled questions.
A chunk is relevant if it contains the question's evidence phrase (for cross-page items, every phrase must be covered). k = 4 (the app's top_k).
Produced by `python retrieval_eval.py`; no LLM calls, fully deterministic.

## Chunking char: A 300/50 (161 chunks, mean 291 chars, max 300)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.41 (16/39) | 0.64 (25/39) | 0.54 | 0.82 | 0.5 |
| MiniLM | 0.36 (14/39) | 0.59 (23/39) | 0.48 | 0.92 | 9.2 |
| Hybrid | 0.44 (17/39) | 0.69 (27/39) | 0.57 | 0.90 | 10.0 |

## Chunking char: Default 800/150 (63 chunks, mean 741 chars, max 800)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.64 (25/39) | 0.87 (34/39) | 0.75 | 0.90 | 0.3 |
| MiniLM | 0.38 (15/39) | 0.77 (30/39) | 0.56 | 0.82 | 10.0 |
| Hybrid | 0.56 (22/39) | 0.82 (32/39) | 0.70 | 0.82 | 10.4 |

## Chunking char: B 1000/200 (52 chunks, mean 902 chars, max 1000)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.56 (22/39) | 0.79 (31/39) | 0.69 | 0.85 | 0.3 |
| MiniLM | 0.44 (17/39) | 0.67 (26/39) | 0.58 | 0.77 | 9.4 |
| Hybrid | 0.54 (21/39) | 0.85 (33/39) | 0.69 | 0.87 | 9.7 |

## Chunking structured: A 300/50 (200 chunks, mean 230 chars, max 300)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.56 (22/39) | 0.74 (29/39) | 0.67 | 0.87 | 0.6 |
| MiniLM | 0.51 (20/39) | 0.77 (30/39) | 0.62 | 0.87 | 9.1 |
| Hybrid | 0.54 (21/39) | 0.85 (33/39) | 0.66 | 0.92 | 10.0 |

Evidence split across chunks (unreachable): [34]

## Chunking structured: Default 800/150 (66 chunks, mean 665 chars, max 800)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.62 (24/39) | 0.90 (35/39) | 0.75 | 0.90 | 0.3 |
| MiniLM | 0.51 (20/39) | 0.77 (30/39) | 0.62 | 0.82 | 9.1 |
| Hybrid | 0.59 (23/39) | 0.82 (32/39) | 0.70 | 0.82 | 9.5 |

## Chunking structured: B 1000/200 (55 chunks, mean 804 chars, max 998)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.64 (25/39) | 0.85 (33/39) | 0.75 | 0.87 | 0.3 |
| MiniLM | 0.54 (21/39) | 0.77 (30/39) | 0.64 | 0.79 | 8.9 |
| Hybrid | 0.62 (24/39) | 0.90 (35/39) | 0.74 | 0.92 | 9.2 |

## Questions missed in the top 4

- **BM25, char: A 300/50** (14): #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #6 Which earlier models used convolutions to reduce sequential computation?; #9 How is the decoder stopped from looking at future tokens?; #12 Why are the dot products divided by the square root of the key dimension?; #15 How many attention heads are used?; #18 Which nonlinearity is used inside the feed-forward network?; #19 How big is the inner layer of the feed-forward network?; #20 How is word order information added to the model?; #32 What happened when learned positional embeddings replaced sinusoids?; #33 How many sentences does the WSJ training set for parsing contain?; #35 Where can the training and evaluation code be found?; #36 What future directions beyond text do the authors mention?; #38 What does the attention visualization for the word making show?
- **Hybrid, char: A 300/50** (12): #2 What BLEU score did the model reach on English to German translation?; #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #5 How quickly can the model reach state of the art translation quality?; #15 How many attention heads are used?; #18 Which nonlinearity is used inside the feed-forward network?; #19 How big is the inner layer of the feed-forward network?; #20 How is word order information added to the model?; #32 What happened when learned positional embeddings replaced sinusoids?; #33 How many sentences does the WSJ training set for parsing contain?; #36 What future directions beyond text do the authors mention?; #37 Who is the author of the Xception paper cited?
- **MiniLM, char: A 300/50** (16): #1 What architecture does the paper propose instead of recurrence and convolutions?; #2 What BLEU score did the model reach on English to German translation?; #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #12 Why are the dot products divided by the square root of the key dimension?; #13 Which two attention functions are most commonly used?; #15 How many attention heads are used?; #16 What is the key and value dimension of each head?; #18 Which nonlinearity is used inside the feed-forward network?; #19 How big is the inner layer of the feed-forward network?; #20 How is word order information added to the model?; #27 What dropout rate did the base model use?; #32 What happened when learned positional embeddings replaced sinusoids?; #36 What future directions beyond text do the authors mention?; #37 Who is the author of the Xception paper cited?; #38 What does the attention visualization for the word making show?
- **BM25, char: B 1000/200** (8): #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #9 How is the decoder stopped from looking at future tokens?; #15 How many attention heads are used?; #16 What is the key and value dimension of each head?; #19 How big is the inner layer of the feed-forward network?; #20 How is word order information added to the model?; #35 Where can the training and evaluation code be found?
- **Hybrid, char: B 1000/200** (6): #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #19 How big is the inner layer of the feed-forward network?; #20 How is word order information added to the model?; #35 Where can the training and evaluation code be found?; #37 Who is the author of the Xception paper cited?
- **MiniLM, char: B 1000/200** (13): #2 What BLEU score did the model reach on English to German translation?; #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #9 How is the decoder stopped from looking at future tokens?; #13 Which two attention functions are most commonly used?; #19 How big is the inner layer of the feed-forward network?; #20 How is word order information added to the model?; #31 How much worse is using a single attention head?; #32 What happened when learned positional embeddings replaced sinusoids?; #33 How many sentences does the WSJ training set for parsing contain?; #35 Where can the training and evaluation code be found?; #36 What future directions beyond text do the authors mention?; #37 Who is the author of the Xception paper cited?
- **BM25, char: Default 800/150** (5): #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #15 How many attention heads are used?; #20 How is word order information added to the model?; #35 Where can the training and evaluation code be found?
- **Hybrid, char: Default 800/150** (7): #1 What architecture does the paper propose instead of recurrence and convolutions?; #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #20 How is word order information added to the model?; #31 How much worse is using a single attention head?; #35 Where can the training and evaluation code be found?; #37 Who is the author of the Xception paper cited?
- **MiniLM, char: Default 800/150** (9): #1 What architecture does the paper propose instead of recurrence and convolutions?; #2 What BLEU score did the model reach on English to German translation?; #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #20 How is word order information added to the model?; #31 How much worse is using a single attention head?; #35 Where can the training and evaluation code be found?; #36 What future directions beyond text do the authors mention?; #37 Who is the author of the Xception paper cited?
- **BM25, structured: A 300/50** (10): #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #8 What is the output size of every sub-layer and embedding layer?; #12 Why are the dot products divided by the square root of the key dimension?; #15 How many attention heads are used?; #18 Which nonlinearity is used inside the feed-forward network?; #20 How is word order information added to the model?; #34 What F1 did the semi-supervised Transformer reach on WSJ section 23?; #35 Where can the training and evaluation code be found?; #36 What future directions beyond text do the authors mention?
- **Hybrid, structured: A 300/50** (6): #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #5 How quickly can the model reach state of the art translation quality?; #20 How is word order information added to the model?; #34 What F1 did the semi-supervised Transformer reach on WSJ section 23?; #36 What future directions beyond text do the authors mention?
- **MiniLM, structured: A 300/50** (9): #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #9 How is the decoder stopped from looking at future tokens?; #13 Which two attention functions are most commonly used?; #15 How many attention heads are used?; #20 How is word order information added to the model?; #34 What F1 did the semi-supervised Transformer reach on WSJ section 23?; #35 Where can the training and evaluation code be found?; #36 What future directions beyond text do the authors mention?
- **BM25, structured: B 1000/200** (6): #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #15 How many attention heads are used?; #16 What is the key and value dimension of each head?; #20 How is word order information added to the model?; #35 Where can the training and evaluation code be found?
- **Hybrid, structured: B 1000/200** (4): #3 Who suggested using self-attention in place of recurrent networks?; #20 How is word order information added to the model?; #35 Where can the training and evaluation code be found?; #37 Who is the author of the Xception paper cited?
- **MiniLM, structured: B 1000/200** (9): #2 What BLEU score did the model reach on English to German translation?; #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #20 How is word order information added to the model?; #25 What hardware were the models trained on?; #31 How much worse is using a single attention head?; #35 Where can the training and evaluation code be found?; #36 What future directions beyond text do the authors mention?; #37 Who is the author of the Xception paper cited?
- **BM25, structured: Default 800/150** (4): #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #15 How many attention heads are used?; #35 Where can the training and evaluation code be found?
- **Hybrid, structured: Default 800/150** (7): #1 What architecture does the paper propose instead of recurrence and convolutions?; #3 Who suggested using self-attention in place of recurrent networks?; #4 Why is it hard to parallelize recurrent models during training?; #15 How many attention heads are used?; #20 How is word order information added to the model?; #35 Where can the training and evaluation code be found?; #37 Who is the author of the Xception paper cited?
- **MiniLM, structured: Default 800/150** (9): #1 What architecture does the paper propose instead of recurrence and convolutions?; #2 What BLEU score did the model reach on English to German translation?; #3 Who suggested using self-attention in place of recurrent networks?; #5 How quickly can the model reach state of the art translation quality?; #20 How is word order information added to the model?; #31 How much worse is using a single attention head?; #35 Where can the training and evaluation code be found?; #36 What future directions beyond text do the authors mention?; #37 Who is the author of the Xception paper cited?

## Chunk embedding time (seconds, CPU)

- MiniLM, char: A 300/50: 2.08
- MiniLM, char: Default 800/150: 1.65
- MiniLM, char: B 1000/200: 1.41
- MiniLM, structured: A 300/50: 1.70
- MiniLM, structured: Default 800/150: 1.59
- MiniLM, structured: B 1000/200: 1.26
