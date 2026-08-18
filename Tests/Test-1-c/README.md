# C Chatbot Language Model

It is intentionally pragmatic:

- The model is a token-level trigram language model with interpolated unigram, bigram, and trigram probabilities.
- The trained artifact also stores dialogue examples so replies can use retrieval-guided matching before falling back to pure n-gram generation.
- The dataset is generated locally by the same C binary and expands into thousands of chatbot training dialogues.
- Training and inference run quickly on a normal terminal without Python or external ML libraries.

## Project Layout

- `src/chatbot_lm.c`: dataset generation, training, model loading, one-shot replies, and interactive chat.
- `data/chat_corpus.txt`: generated training corpus.
- `model/chatbot.model`: trained model artifact.
- `bin/chatbot_lm`: compiled binary.

## Build

```sh
cd BIRO-AI/Tests/Test-1-c
make
```

## Generate Training Data

```sh
./bin/chatbot_lm generate-data data/chat_corpus.txt
```

The generated corpus contains thousands of synthetic chat dialogues covering:

- greetings and identity prompts
- capability questions
- C programming requests
- debugging guidance
- writing and rewriting tasks
- study plans
- summaries
- math explanations
- productivity support
- translation-style prompts

## Train The Model

```sh
./bin/chatbot_lm train data/chat_corpus.txt model/chatbot.model
```

## Use The Model

One-shot reply:

```sh
./bin/chatbot_lm reply model/chatbot.model "help me debug a segfault in c"
```

Interactive mode:

```sh
./bin/chatbot_lm chat model/chatbot.model
```

Type `exit` or `quit` to stop the interactive session.

## Notes

- This is a classical n-gram language model with retrieval-guided reply selection, not a transformer.
- It is appropriate for a lightweight chatbot demo, fast local training, and readable C code.
- Because the model is small and the corpus is synthetic, responses are best on short practical prompts similar to the bundled training data.
