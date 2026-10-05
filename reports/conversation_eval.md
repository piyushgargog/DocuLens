# Conversational retrieval evaluation

23 follow-up questions on one document, 66 chunks (structured chunker). No LLM calls. 'resolved' is the deterministic resolver only; the LLM rewrite is reserved for harder cases and is not scored here.


## Follow-ups (pronouns, ellipsis)

| Strategy | Hit@4 | MRR@10 | Page-Hit@4 | n |
|---|---|---|---|---|
| alone | 0.67 | 0.49 | 0.73 | 15 |
| concat (old) | 0.87 | 0.68 | 0.93 | 15 |
| resolved | 0.93 | 0.73 | 0.87 | 15 |
| standalone | 0.93 | 0.81 | 1.00 | 15 |

## Topic changes (self-contained question after an unrelated one)

| Strategy | Hit@4 | MRR@10 | Page-Hit@4 | n |
|---|---|---|---|---|
| alone | 1.00 | 1.00 | 1.00 | 8 |
| concat (old) | 1.00 | 1.00 | 1.00 | 8 |
| resolved | 1.00 | 1.00 | 1.00 | 8 |
| standalone | 1.00 | 1.00 | 1.00 | 8 |

## Per question (Hit@4: alone / old concatenation / resolved)

- [follow_up] #1 What is the dimension of each of them? -> `What is the dimension of each of them? attention heads Transformer` (deterministic): 1 / 1 / 1
- [follow_up] #2 What were its hyperparameters? -> `What were its hyperparameters? optimizer training` (deterministic): 0 / 1 / 1
- [follow_up] #3 Why did the authors choose that version? -> `Why did the authors choose that version? positional encodings` (deterministic): 1 / 1 / 1
- [follow_up] #4 How big is it? -> `How big is it? dataset English-German` (deterministic): 0 / 0 / 1
- [follow_up] #5 How long did it train? -> `How long did the big model train?` (deterministic): 0 / 1 / 1
- [follow_up] #6 Why are the dot products scaled? -> `Why are the dot products scaled? dot-product attention` (deterministic): 1 / 1 / 1
- [follow_up] #7 What third sub-layer does it add? -> `What third sub-layer does it add? decoder contain` (deterministic): 1 / 1 / 1
- [follow_up] #8 How large is its inner layer? -> `How large is its inner layer? activation feed-forward network` (deterministic): 1 / 1 / 1
- [follow_up] #9 And what dropout rate did the base model use? -> `And what dropout rate did the base model use? regularized` (deterministic): 1 / 1 / 1
- [follow_up] #10 How does that compare with a recurrent layer? -> `How does that compare with a recurrent layer? self-attention layer's maximum path length` (deterministic): 1 / 1 / 1
- [follow_up] #11 How long did each step take for them? -> `How long did each step take for them? GPUs training` (deterministic): 1 / 1 / 1
- [follow_up] #12 And the length penalty? -> `And the length penalty? beam search settings` (deterministic): 1 / 1 / 1
- [follow_up] #13 Does it help or hurt? -> `Does label smoothing help or hurt?` (deterministic): 0 / 1 / 1
- [follow_up] #14 Which one gave better results? -> `Which one gave better results?` (standalone): 1 / 1 / 1
- [follow_up] #15 What happens if only one is used? -> `What happens if only one is used?` (standalone): 0 / 0 / 0
- [topic_change] #16 What hardware were the models trained on? -> `What hardware were the models trained on?` (standalone): 1 / 1 / 1
- [topic_change] #17 What is the beam size? -> `What is the beam size?` (standalone): 1 / 1 / 1
- [topic_change] #18 What dropout rate did the base model use? -> `What dropout rate did the base model use?` (standalone): 1 / 1 / 1
- [topic_change] #19 How many warmup steps are used? -> `How many warmup steps are used?` (standalone): 1 / 1 / 1
- [topic_change] #20 What activation does the feed-forward network use? -> `What activation does the feed-forward network use?` (standalone): 1 / 1 / 1
- [topic_change] #21 Which label smoothing value was used? -> `Which label smoothing value was used?` (standalone): 1 / 1 / 1
- [topic_change] #22 What vocabulary was used for English-French? -> `What vocabulary was used for English-French?` (standalone): 1 / 1 / 1
- [topic_change] #23 How many sentence pairs are in the English-German training data? -> `How many sentence pairs are in the English-German training data?` (standalone): 1 / 1 / 1

Limitations: 15 questions, one document, written by the author; the previous question is always the one the follow-up refers to (no multi-turn drift, no references to earlier answers).
