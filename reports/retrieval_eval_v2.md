# Retrieval Evaluation

Document: `sample_docs/dev_real_world_document.pdf` — 37 labelled questions.
A chunk is relevant if it contains the question's evidence phrase (for cross-page items, every phrase must be covered). k = 4 (the app's top_k).
Produced by `python retrieval_eval.py`; no LLM calls, fully deterministic.

## Chunking char: A 300/50 (161 chunks, mean 291 chars, max 300)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.43 (16/37) | 0.65 (24/37) | 0.52 | 0.89 | 0.5 |
| MiniLM | 0.46 (17/37) | 0.59 (22/37) | 0.52 | 0.89 | 9.9 |
| Hybrid | 0.41 (15/37) | 0.54 (20/37) | 0.50 | 0.84 | 10.6 |
| Hybrid-HNSW | 0.41 (15/37) | 0.54 (20/37) | 0.50 | 0.84 | 10.7 |

By category, Hit@4 (n = questions):

| Category | n | BM25 | MiniLM | Hybrid | Hybrid-HNSW |
|---|---|---|---|---|---|
| cross-page | 3 | 0.00 | 0.33 | 0.00 | 0.00 |
| entity | 4 | 0.75 | 0.50 | 0.50 | 0.50 |
| number | 7 | 1.00 | 0.86 | 0.86 | 0.86 |
| paraphrase | 10 | 0.60 | 0.70 | 0.60 | 0.60 |
| short | 7 | 0.71 | 0.71 | 0.57 | 0.57 |
| table | 6 | 0.50 | 0.17 | 0.33 | 0.33 |

## Chunking char: Default 800/150 (63 chunks, mean 741 chars, max 800)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.49 (18/37) | 0.81 (30/37) | 0.61 | 0.84 | 0.3 |
| MiniLM | 0.51 (19/37) | 0.68 (25/37) | 0.60 | 0.76 | 9.5 |
| Hybrid | 0.51 (19/37) | 0.73 (27/37) | 0.62 | 0.78 | 9.9 |
| Hybrid-HNSW | 0.51 (19/37) | 0.73 (27/37) | 0.62 | 0.78 | 10.0 |

By category, Hit@4 (n = questions):

| Category | n | BM25 | MiniLM | Hybrid | Hybrid-HNSW |
|---|---|---|---|---|---|
| cross-page | 3 | 0.33 | 0.00 | 0.33 | 0.33 |
| entity | 4 | 0.75 | 0.25 | 0.50 | 0.50 |
| number | 7 | 1.00 | 0.86 | 1.00 | 1.00 |
| paraphrase | 10 | 0.80 | 0.60 | 0.50 | 0.50 |
| short | 7 | 0.86 | 1.00 | 1.00 | 1.00 |
| table | 6 | 0.83 | 0.83 | 0.83 | 0.83 |

## Chunking char: B 1000/200 (52 chunks, mean 902 chars, max 1000)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.51 (19/37) | 0.78 (29/37) | 0.63 | 0.81 | 0.3 |
| MiniLM | 0.57 (21/37) | 0.62 (23/37) | 0.61 | 0.70 | 11.1 |
| Hybrid | 0.54 (20/37) | 0.76 (28/37) | 0.64 | 0.81 | 11.5 |
| Hybrid-HNSW | 0.54 (20/37) | 0.76 (28/37) | 0.64 | 0.81 | 12.0 |

By category, Hit@4 (n = questions):

| Category | n | BM25 | MiniLM | Hybrid | Hybrid-HNSW |
|---|---|---|---|---|---|
| cross-page | 3 | 0.33 | 0.00 | 0.33 | 0.33 |
| entity | 4 | 0.75 | 0.25 | 0.50 | 0.50 |
| number | 7 | 1.00 | 0.71 | 1.00 | 1.00 |
| paraphrase | 10 | 0.70 | 0.60 | 0.60 | 0.60 |
| short | 7 | 0.86 | 0.86 | 1.00 | 1.00 |
| table | 6 | 0.83 | 0.83 | 0.83 | 0.83 |

## Chunking structured: A 300/50 (200 chunks, mean 230 chars, max 300)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.46 (17/37) | 0.57 (21/37) | 0.53 | 0.86 | 0.6 |
| MiniLM | 0.38 (14/37) | 0.54 (20/37) | 0.48 | 0.86 | 11.6 |
| Hybrid | 0.46 (17/37) | 0.59 (22/37) | 0.54 | 0.89 | 12.5 |
| Hybrid-HNSW | 0.46 (17/37) | 0.59 (22/37) | 0.54 | 0.89 | 12.8 |

Evidence split across chunks (unreachable): [23]

By category, Hit@4 (n = questions):

| Category | n | BM25 | MiniLM | Hybrid | Hybrid-HNSW |
|---|---|---|---|---|---|
| cross-page | 3 | 0.00 | 0.33 | 0.00 | 0.00 |
| entity | 4 | 0.50 | 0.50 | 0.75 | 0.75 |
| number | 7 | 1.00 | 0.71 | 1.00 | 1.00 |
| paraphrase | 10 | 0.50 | 0.70 | 0.50 | 0.50 |
| short | 7 | 0.71 | 0.57 | 0.71 | 0.71 |
| table | 6 | 0.33 | 0.17 | 0.33 | 0.33 |

## Chunking structured: Default 800/150 (66 chunks, mean 665 chars, max 800)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.54 (20/37) | 0.73 (27/37) | 0.62 | 0.86 | 0.4 |
| MiniLM | 0.51 (19/37) | 0.78 (29/37) | 0.64 | 0.81 | 13.6 |
| Hybrid | 0.62 (23/37) | 0.78 (29/37) | 0.70 | 0.81 | 14.1 |
| Hybrid-HNSW | 0.62 (23/37) | 0.78 (29/37) | 0.70 | 0.81 | 12.5 |

By category, Hit@4 (n = questions):

| Category | n | BM25 | MiniLM | Hybrid | Hybrid-HNSW |
|---|---|---|---|---|---|
| cross-page | 3 | 0.33 | 0.33 | 0.33 | 0.33 |
| entity | 4 | 0.75 | 0.25 | 0.50 | 0.50 |
| number | 7 | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrase | 10 | 0.50 | 0.80 | 0.70 | 0.70 |
| short | 7 | 0.86 | 1.00 | 1.00 | 1.00 |
| table | 6 | 0.83 | 0.83 | 0.83 | 0.83 |

## Chunking structured: B 1000/200 (55 chunks, mean 804 chars, max 998)

| Retriever | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.46 (17/37) | 0.76 (28/37) | 0.60 | 0.84 | 0.3 |
| MiniLM | 0.49 (18/37) | 0.76 (28/37) | 0.59 | 0.86 | 10.5 |
| Hybrid | 0.57 (21/37) | 0.76 (28/37) | 0.67 | 0.86 | 10.9 |
| Hybrid-HNSW | 0.57 (21/37) | 0.76 (28/37) | 0.67 | 0.86 | 10.6 |

By category, Hit@4 (n = questions):

| Category | n | BM25 | MiniLM | Hybrid | Hybrid-HNSW |
|---|---|---|---|---|---|
| cross-page | 3 | 0.00 | 0.33 | 0.33 | 0.33 |
| entity | 4 | 0.75 | 0.25 | 0.50 | 0.50 |
| number | 7 | 1.00 | 0.86 | 1.00 | 1.00 |
| paraphrase | 10 | 0.70 | 0.80 | 0.60 | 0.60 |
| short | 7 | 0.86 | 1.00 | 1.00 | 1.00 |
| table | 6 | 0.83 | 0.83 | 0.83 | 0.83 |

## Questions missed in the top 4

- **BM25, char: A 300/50** (13): #1 Which regularisation technique randomly silences units, and where is it applied?; #5 Why does splitting into several heads not make it more expensive?; #8 What ability does attending in parallel give that one head lacks?; #10 What settings does the Adam optimiser run with?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #30 Positional encoding; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid, char: A 300/50** (17): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #9 Do the input and output word tables share parameters?; #16 What is the size of the inner feed-forward layer?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #27 How much did the Deep-Att ensemble cost to train for English-French?; #28 Dropout rate?; #30 Positional encoding; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid-HNSW, char: A 300/50** (17): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #9 Do the input and output word tables share parameters?; #16 What is the size of the inner feed-forward layer?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #27 How much did the Deep-Att ensemble cost to train for English-French?; #28 Dropout rate?; #30 Positional encoding; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **MiniLM, char: A 300/50** (15): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #9 Do the input and output word tables share parameters?; #16 What is the size of the inner feed-forward layer?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #23 What is the per-layer complexity of a convolutional layer?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #27 How much did the Deep-Att ensemble cost to train for English-French?; #28 Dropout rate?; #30 Positional encoding; #35 How is multi-head attention built and what dimensions does each head get?; #37 What did the big model score on English-German and how long did it take to train?
- **BM25, char: B 1000/200** (8): #1 Which regularisation technique randomly silences units, and where is it applied?; #5 Why does splitting into several heads not make it more expensive?; #8 What ability does attending in parallel give that one head lacks?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #31 Number of heads?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid, char: B 1000/200** (9): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #6 What makes the model easier to inspect and understand?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid-HNSW, char: B 1000/200** (9): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #6 What makes the model easier to inspect and understand?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **MiniLM, char: B 1000/200** (14): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #6 What makes the model easier to inspect and understand?; #9 Do the input and output word tables share parameters?; #14 What is the value of the length penalty?; #16 What is the size of the inner feed-forward layer?; #18 Which author is from the University of Toronto?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #29 Beam size?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **BM25, char: Default 800/150** (7): #1 Which regularisation technique randomly silences units, and where is it applied?; #5 Why does splitting into several heads not make it more expensive?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #31 Number of heads?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid, char: Default 800/150** (10): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #6 What makes the model easier to inspect and understand?; #9 Do the input and output word tables share parameters?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid-HNSW, char: Default 800/150** (10): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #6 What makes the model easier to inspect and understand?; #9 Do the input and output word tables share parameters?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **MiniLM, char: Default 800/150** (12): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #6 What makes the model easier to inspect and understand?; #9 Do the input and output word tables share parameters?; #16 What is the size of the inner feed-forward layer?; #18 Which author is from the University of Toronto?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **BM25, structured: A 300/50** (16): #1 Which regularisation technique randomly silences units, and where is it applied?; #2 What stops the decoder from peeking at words that come later?; #5 Why does splitting into several heads not make it more expensive?; #6 What makes the model easier to inspect and understand?; #8 What ability does attending in parallel give that one head lacks?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #23 What is the per-layer complexity of a convolutional layer?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #30 Positional encoding; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid, structured: A 300/50** (15): #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #6 What makes the model easier to inspect and understand?; #8 What ability does attending in parallel give that one head lacks?; #9 Do the input and output word tables share parameters?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #23 What is the per-layer complexity of a convolutional layer?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #30 Positional encoding; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid-HNSW, structured: A 300/50** (15): #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #6 What makes the model easier to inspect and understand?; #8 What ability does attending in parallel give that one head lacks?; #9 Do the input and output word tables share parameters?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #23 What is the per-layer complexity of a convolutional layer?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #30 Positional encoding; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **MiniLM, structured: A 300/50** (17): #4 Why can order information simply be summed with the word vectors?; #6 What makes the model easier to inspect and understand?; #9 Do the input and output word tables share parameters?; #14 What is the value of the length penalty?; #15 How many of the most recent checkpoints were averaged for the base models?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #22 What BLEU scores does the big Transformer reach on EN-DE and EN-FR?; #23 What is the per-layer complexity of a convolutional layer?; #25 What dev BLEU and perplexity does the big configuration get in the variations table?; #26 What happens in the ablation with a single attention head?; #27 How much did the Deep-Att ensemble cost to train for English-French?; #28 Dropout rate?; #30 Positional encoding; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #37 What did the big model score on English-German and how long did it take to train?
- **BM25, structured: B 1000/200** (9): #1 Which regularisation technique randomly silences units, and where is it applied?; #5 Why does splitting into several heads not make it more expensive?; #8 What ability does attending in parallel give that one head lacks?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #31 Number of heads?; #35 How is multi-head attention built and what dimensions does each head get?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid, structured: B 1000/200** (9): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #9 Do the input and output word tables share parameters?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid-HNSW, structured: B 1000/200** (9): #1 Which regularisation technique randomly silences units, and where is it applied?; #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #9 Do the input and output word tables share parameters?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **MiniLM, structured: B 1000/200** (9): #4 Why can order information simply be summed with the word vectors?; #9 Do the input and output word tables share parameters?; #16 What is the size of the inner feed-forward layer?; #18 Which author is from the University of Toronto?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **BM25, structured: Default 800/150** (10): #1 Which regularisation technique randomly silences units, and where is it applied?; #2 What stops the decoder from peeking at words that come later?; #5 Why does splitting into several heads not make it more expensive?; #7 Under what condition is attention cheaper than a recurrent layer?; #8 What ability does attending in parallel give that one head lacks?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #31 Number of heads?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid, structured: Default 800/150** (8): #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #9 Do the input and output word tables share parameters?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **Hybrid-HNSW, structured: Default 800/150** (8): #4 Why can order information simply be summed with the word vectors?; #5 Why does splitting into several heads not make it more expensive?; #9 Do the input and output word tables share parameters?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?
- **MiniLM, structured: Default 800/150** (8): #4 Why can order information simply be summed with the word vectors?; #9 Do the input and output word tables share parameters?; #18 Which author is from the University of Toronto?; #19 Who wrote the Xception paper that is cited?; #20 Which earlier translation system used reinforcement learning, per the results table?; #26 What happens in the ablation with a single attention head?; #36 What regularisation is used, and what dropout did the big English-French model use?; #37 What did the big model score on English-German and how long did it take to train?

## Chunk embedding time (seconds, CPU)

- MiniLM, char: A 300/50: 2.33
- MiniLM, char: Default 800/150: 1.85
- MiniLM, char: B 1000/200: 1.53
- MiniLM, structured: A 300/50: 2.33
- MiniLM, structured: Default 800/150: 1.86
- MiniLM, structured: B 1000/200: 1.58
