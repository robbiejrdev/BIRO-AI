CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
LDLIBS ?= -lm

BIN := bin/chatbot_lm
SRC := src/chatbot_lm.c
CORPUS := data/chat_corpus.txt
MODEL := model/chatbot.model

.PHONY: all clean train demo

all: $(BIN)

$(BIN): $(SRC)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

$(CORPUS): $(BIN)
	./$(BIN) generate-data $(CORPUS)

$(MODEL): $(BIN) $(CORPUS)
	./$(BIN) train $(CORPUS) $(MODEL)

train: $(MODEL)

demo: $(MODEL)
	./$(BIN) reply $(MODEL) "hello can you help me write a c function ?"

clean:
	rm -f $(BIN) $(MODEL) $(CORPUS)
