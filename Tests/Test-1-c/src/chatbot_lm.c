#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARRAY_LEN(x) ((int)(sizeof(x) / sizeof((x)[0])))
#define DEFAULT_MAX_REPLY_TOKENS 48
#define DEFAULT_TEMPERATURE 0.82
#define TOP_K 12

typedef struct {
    char *key;
    int value;
} VocabSlot;

typedef struct {
    char **tokens;
    int size;
    int cap;
    VocabSlot *slots;
    int slot_cap;
} Vocabulary;

typedef struct {
    int *data;
    int size;
    int cap;
} IntVector;

typedef struct {
    char *user_text;
    char *user_norm;
    char *assistant_text;
} DialogueExample;

typedef struct {
    DialogueExample *items;
    int size;
    int cap;
} ExampleSet;

typedef struct {
    int a;
    int b;
    int count;
    unsigned char used;
} BigramEntry;

typedef struct {
    BigramEntry *entries;
    int cap;
    int size;
} BigramTable;

typedef struct {
    int a;
    int b;
    int c;
    int count;
    unsigned char used;
} TrigramEntry;

typedef struct {
    TrigramEntry *entries;
    int cap;
    int size;
} TrigramTable;

typedef struct {
    Vocabulary vocab;
    int *unigrams;
    int unigram_cap;
    long long total_tokens;
    BigramTable bigrams;
    TrigramTable trigrams;
    int bos_id;
    int user_id;
    int assistant_id;
    int end_id;
    int unk_id;
    ExampleSet examples;
} Model;

typedef struct {
    int id;
    double score;
} Candidate;

static char *normalize_text(const char *input);
static int is_punctuation_token(const char *token);
static int is_terminal_punctuation_token(const char *token);

static void die(const char *fmt, ...) {
    va_list args;
    fprintf(stderr, "error: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        die("out of memory");
    }
    return ptr;
}

static void *xcalloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (!ptr) {
        die("out of memory");
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (!next) {
        die("out of memory");
    }
    return next;
}

static char *xstrdup(const char *src) {
    size_t len = strlen(src);
    char *copy = xmalloc(len + 1);
    memcpy(copy, src, len + 1);
    return copy;
}

static int next_power_of_two(int value) {
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

static uint64_t hash_string(const char *s) {
    uint64_t hash = 1469598103934665603ull;
    while (*s) {
        hash ^= (unsigned char) *s++;
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t mix_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

static uint64_t hash_bigram_key(int a, int b) {
    uint64_t key = ((uint64_t) (unsigned int) a << 32) ^ (unsigned int) b;
    return mix_u64(key);
}

static uint64_t hash_trigram_key(int a, int b, int c) {
    uint64_t key = ((uint64_t) (unsigned int) a << 42)
        ^ ((uint64_t) (unsigned int) b << 21)
        ^ (unsigned int) c;
    return mix_u64(key);
}

static void vocab_init(Vocabulary *vocab, int initial_cap) {
    vocab->size = 0;
    vocab->cap = initial_cap > 0 ? initial_cap : 64;
    vocab->tokens = xcalloc((size_t) vocab->cap, sizeof(char *));
    vocab->slot_cap = next_power_of_two(vocab->cap * 4);
    vocab->slots = xcalloc((size_t) vocab->slot_cap, sizeof(VocabSlot));
}

static void vocab_free(Vocabulary *vocab) {
    int i;
    if (!vocab) {
        return;
    }
    for (i = 0; i < vocab->size; i++) {
        free(vocab->tokens[i]);
    }
    free(vocab->tokens);
    free(vocab->slots);
    memset(vocab, 0, sizeof(*vocab));
}

static void vocab_rehash(Vocabulary *vocab, int new_slot_cap) {
    int i;
    VocabSlot *old_slots = vocab->slots;
    int old_cap = vocab->slot_cap;
    vocab->slot_cap = next_power_of_two(new_slot_cap);
    vocab->slots = xcalloc((size_t) vocab->slot_cap, sizeof(VocabSlot));
    for (i = 0; i < old_cap; i++) {
        if (old_slots[i].key) {
            uint64_t hash = hash_string(old_slots[i].key);
            int mask = vocab->slot_cap - 1;
            int idx = (int) (hash & (uint64_t) mask);
            while (vocab->slots[idx].key) {
                idx = (idx + 1) & mask;
            }
            vocab->slots[idx] = old_slots[i];
        }
    }
    free(old_slots);
}

static int vocab_lookup_id(const Vocabulary *vocab, const char *token) {
    uint64_t hash;
    int mask;
    int idx;
    if (vocab->slot_cap == 0) {
        return -1;
    }
    hash = hash_string(token);
    mask = vocab->slot_cap - 1;
    idx = (int) (hash & (uint64_t) mask);
    while (vocab->slots[idx].key) {
        if (strcmp(vocab->slots[idx].key, token) == 0) {
            return vocab->slots[idx].value;
        }
        idx = (idx + 1) & mask;
    }
    return -1;
}

static int vocab_get_or_add(Vocabulary *vocab, const char *token) {
    uint64_t hash;
    int mask;
    int idx;
    int id;
    if ((vocab->size + 1) * 10 > vocab->slot_cap * 7) {
        vocab_rehash(vocab, vocab->slot_cap * 2);
    }

    hash = hash_string(token);
    mask = vocab->slot_cap - 1;
    idx = (int) (hash & (uint64_t) mask);
    while (vocab->slots[idx].key) {
        if (strcmp(vocab->slots[idx].key, token) == 0) {
            return vocab->slots[idx].value;
        }
        idx = (idx + 1) & mask;
    }

    if (vocab->size == vocab->cap) {
        vocab->cap *= 2;
        vocab->tokens = xrealloc(vocab->tokens, (size_t) vocab->cap * sizeof(char *));
    }

    id = vocab->size++;
    vocab->tokens[id] = xstrdup(token);
    vocab->slots[idx].key = vocab->tokens[id];
    vocab->slots[idx].value = id;
    return id;
}

static void int_vector_init(IntVector *vec) {
    vec->size = 0;
    vec->cap = 32;
    vec->data = xmalloc((size_t) vec->cap * sizeof(int));
}

static void int_vector_push(IntVector *vec, int value) {
    if (vec->size == vec->cap) {
        vec->cap *= 2;
        vec->data = xrealloc(vec->data, (size_t) vec->cap * sizeof(int));
    }
    vec->data[vec->size++] = value;
}

static void int_vector_clear(IntVector *vec) {
    vec->size = 0;
}

static void int_vector_free(IntVector *vec) {
    free(vec->data);
    memset(vec, 0, sizeof(*vec));
}

static void example_set_init(ExampleSet *set) {
    set->size = 0;
    set->cap = 128;
    set->items = xcalloc((size_t) set->cap, sizeof(DialogueExample));
}

static void example_set_add(ExampleSet *set, const char *user_text, const char *assistant_text) {
    DialogueExample *example;
    if (!user_text || !assistant_text || user_text[0] == '\0' || assistant_text[0] == '\0') {
        return;
    }
    if (set->size == set->cap) {
        set->cap *= 2;
        set->items = xrealloc(set->items, (size_t) set->cap * sizeof(DialogueExample));
    }
    example = &set->items[set->size++];
    example->user_text = xstrdup(user_text);
    example->user_norm = normalize_text(user_text);
    example->assistant_text = xstrdup(assistant_text);
}

static void example_set_free(ExampleSet *set) {
    int i;
    for (i = 0; i < set->size; i++) {
        free(set->items[i].user_text);
        free(set->items[i].user_norm);
        free(set->items[i].assistant_text);
    }
    free(set->items);
    memset(set, 0, sizeof(*set));
}

static void bigram_init(BigramTable *table, int initial_cap) {
    table->cap = next_power_of_two(initial_cap > 0 ? initial_cap : 256);
    table->size = 0;
    table->entries = xcalloc((size_t) table->cap, sizeof(BigramEntry));
}

static void bigram_free(BigramTable *table) {
    free(table->entries);
    memset(table, 0, sizeof(*table));
}

static void bigram_insert_raw(BigramTable *table, BigramEntry entry) {
    int mask = table->cap - 1;
    int idx = (int) (hash_bigram_key(entry.a, entry.b) & (uint64_t) mask);
    while (table->entries[idx].used) {
        idx = (idx + 1) & mask;
    }
    table->entries[idx] = entry;
}

static void bigram_rehash(BigramTable *table, int new_cap) {
    BigramEntry *old_entries = table->entries;
    int old_cap = table->cap;
    int i;
    table->cap = next_power_of_two(new_cap);
    table->entries = xcalloc((size_t) table->cap, sizeof(BigramEntry));
    table->size = 0;
    for (i = 0; i < old_cap; i++) {
        if (old_entries[i].used) {
            bigram_insert_raw(table, old_entries[i]);
            table->size++;
        }
    }
    free(old_entries);
}

static void bigram_add(BigramTable *table, int a, int b, int delta) {
    int mask;
    int idx;
    if ((table->size + 1) * 10 > table->cap * 7) {
        bigram_rehash(table, table->cap * 2);
    }
    mask = table->cap - 1;
    idx = (int) (hash_bigram_key(a, b) & (uint64_t) mask);
    while (table->entries[idx].used) {
        if (table->entries[idx].a == a && table->entries[idx].b == b) {
            table->entries[idx].count += delta;
            return;
        }
        idx = (idx + 1) & mask;
    }
    table->entries[idx].used = 1;
    table->entries[idx].a = a;
    table->entries[idx].b = b;
    table->entries[idx].count = delta;
    table->size++;
}

static int bigram_get(const BigramTable *table, int a, int b) {
    int mask;
    int idx;
    if (table->cap == 0) {
        return 0;
    }
    mask = table->cap - 1;
    idx = (int) (hash_bigram_key(a, b) & (uint64_t) mask);
    while (table->entries[idx].used) {
        if (table->entries[idx].a == a && table->entries[idx].b == b) {
            return table->entries[idx].count;
        }
        idx = (idx + 1) & mask;
    }
    return 0;
}

static void trigram_init(TrigramTable *table, int initial_cap) {
    table->cap = next_power_of_two(initial_cap > 0 ? initial_cap : 1024);
    table->size = 0;
    table->entries = xcalloc((size_t) table->cap, sizeof(TrigramEntry));
}

static void trigram_free(TrigramTable *table) {
    free(table->entries);
    memset(table, 0, sizeof(*table));
}

static void trigram_insert_raw(TrigramTable *table, TrigramEntry entry) {
    int mask = table->cap - 1;
    int idx = (int) (hash_trigram_key(entry.a, entry.b, entry.c) & (uint64_t) mask);
    while (table->entries[idx].used) {
        idx = (idx + 1) & mask;
    }
    table->entries[idx] = entry;
}

static void trigram_rehash(TrigramTable *table, int new_cap) {
    TrigramEntry *old_entries = table->entries;
    int old_cap = table->cap;
    int i;
    table->cap = next_power_of_two(new_cap);
    table->entries = xcalloc((size_t) table->cap, sizeof(TrigramEntry));
    table->size = 0;
    for (i = 0; i < old_cap; i++) {
        if (old_entries[i].used) {
            trigram_insert_raw(table, old_entries[i]);
            table->size++;
        }
    }
    free(old_entries);
}

static void trigram_add(TrigramTable *table, int a, int b, int c, int delta) {
    int mask;
    int idx;
    if ((table->size + 1) * 10 > table->cap * 7) {
        trigram_rehash(table, table->cap * 2);
    }
    mask = table->cap - 1;
    idx = (int) (hash_trigram_key(a, b, c) & (uint64_t) mask);
    while (table->entries[idx].used) {
        if (table->entries[idx].a == a && table->entries[idx].b == b && table->entries[idx].c == c) {
            table->entries[idx].count += delta;
            return;
        }
        idx = (idx + 1) & mask;
    }
    table->entries[idx].used = 1;
    table->entries[idx].a = a;
    table->entries[idx].b = b;
    table->entries[idx].c = c;
    table->entries[idx].count = delta;
    table->size++;
}

static int trigram_get(const TrigramTable *table, int a, int b, int c) {
    int mask;
    int idx;
    if (table->cap == 0) {
        return 0;
    }
    mask = table->cap - 1;
    idx = (int) (hash_trigram_key(a, b, c) & (uint64_t) mask);
    while (table->entries[idx].used) {
        if (table->entries[idx].a == a && table->entries[idx].b == b && table->entries[idx].c == c) {
            return table->entries[idx].count;
        }
        idx = (idx + 1) & mask;
    }
    return 0;
}

static void model_init(Model *model) {
    memset(model, 0, sizeof(*model));
    vocab_init(&model->vocab, 256);
    bigram_init(&model->bigrams, 4096);
    trigram_init(&model->trigrams, 16384);
    example_set_init(&model->examples);
    model->unigram_cap = 256;
    model->unigrams = xcalloc((size_t) model->unigram_cap, sizeof(int));
    model->bos_id = -1;
    model->user_id = -1;
    model->assistant_id = -1;
    model->end_id = -1;
    model->unk_id = -1;
}

static void model_free(Model *model) {
    if (!model) {
        return;
    }
    vocab_free(&model->vocab);
    free(model->unigrams);
    bigram_free(&model->bigrams);
    trigram_free(&model->trigrams);
    example_set_free(&model->examples);
    memset(model, 0, sizeof(*model));
}

static void model_ensure_unigrams(Model *model, int token_id) {
    int old_cap;
    if (token_id < model->unigram_cap) {
        return;
    }
    old_cap = model->unigram_cap;
    while (token_id >= model->unigram_cap) {
        model->unigram_cap *= 2;
    }
    model->unigrams = xrealloc(model->unigrams, (size_t) model->unigram_cap * sizeof(int));
    memset(model->unigrams + old_cap, 0, (size_t) (model->unigram_cap - old_cap) * sizeof(int));
}

static int model_get_or_add_token(Model *model, const char *token) {
    int id = vocab_get_or_add(&model->vocab, token);
    model_ensure_unigrams(model, id);
    return id;
}

static void model_add_special_tokens(Model *model) {
    model->bos_id = model_get_or_add_token(model, "<bos>");
    model->user_id = model_get_or_add_token(model, "<user>");
    model->assistant_id = model_get_or_add_token(model, "<assistant>");
    model->end_id = model_get_or_add_token(model, "<end>");
    model->unk_id = model_get_or_add_token(model, "<unk>");
}

static void model_resolve_special_ids(Model *model) {
    model->bos_id = vocab_lookup_id(&model->vocab, "<bos>");
    model->user_id = vocab_lookup_id(&model->vocab, "<user>");
    model->assistant_id = vocab_lookup_id(&model->vocab, "<assistant>");
    model->end_id = vocab_lookup_id(&model->vocab, "<end>");
    model->unk_id = vocab_lookup_id(&model->vocab, "<unk>");
    if (model->bos_id < 0 || model->user_id < 0 || model->assistant_id < 0
        || model->end_id < 0 || model->unk_id < 0) {
        die("model is missing one or more required special tokens");
    }
}

static int is_blank_line(const char *line) {
    while (*line) {
        if (!isspace((unsigned char) *line)) {
            return 0;
        }
        line++;
    }
    return 1;
}

static char *detokenize_token_span(char **tokens, int start, int end) {
    size_t cap = 256;
    size_t len = 0;
    char *output = xmalloc(cap);
    int i;
    int capitalize_next = 1;

    output[0] = '\0';
    for (i = start; i < end; i++) {
        const char *token = tokens[i];
        size_t token_len = strlen(token);
        if (len + token_len + 3 >= cap) {
            while (len + token_len + 3 >= cap) {
                cap *= 2;
            }
            output = xrealloc(output, cap);
        }
        if (is_punctuation_token(token)) {
            if (len > 0 && output[len - 1] == ' ') {
                len--;
            }
            memcpy(output + len, token, token_len);
            len += token_len;
            output[len++] = ' ';
            output[len] = '\0';
            if (is_terminal_punctuation_token(token)) {
                capitalize_next = 1;
            }
            continue;
        }
        if (len > 0 && output[len - 1] != ' ') {
            output[len++] = ' ';
        }
        memcpy(output + len, token, token_len);
        if (capitalize_next) {
            output[len] = (char) toupper((unsigned char) output[len]);
            capitalize_next = 0;
        }
        len += token_len;
        output[len++] = ' ';
        output[len] = '\0';
    }
    while (len > 0 && output[len - 1] == ' ') {
        len--;
    }
    output[len] = '\0';
    return output;
}

static void model_add_examples_from_tokens(Model *model, char **tokens, int token_count) {
    int i = 0;
    while (i < token_count) {
        if (strcmp(tokens[i], "<user>") == 0) {
            int assistant_idx = -1;
            int response_end;
            char *user_text;
            char *assistant_text;
            int j;

            for (j = i + 1; j < token_count; j++) {
                if (strcmp(tokens[j], "<assistant>") == 0) {
                    assistant_idx = j;
                    break;
                }
                if (strcmp(tokens[j], "<user>") == 0 || strcmp(tokens[j], "<end>") == 0) {
                    break;
                }
            }
            if (assistant_idx < 0) {
                break;
            }

            response_end = assistant_idx + 1;
            while (response_end < token_count
                   && strcmp(tokens[response_end], "<user>") != 0
                   && strcmp(tokens[response_end], "<end>") != 0) {
                response_end++;
            }

            if (assistant_idx > i + 1 && response_end > assistant_idx + 1) {
                user_text = detokenize_token_span(tokens, i + 1, assistant_idx);
                assistant_text = detokenize_token_span(tokens, assistant_idx + 1, response_end);
                example_set_add(&model->examples, user_text, assistant_text);
                free(user_text);
                free(assistant_text);
            }

            i = response_end;
        } else {
            i++;
        }
    }
}

static void model_count_sequence(Model *model, const IntVector *seq) {
    int i;
    for (i = 0; i < seq->size; i++) {
        int token = seq->data[i];
        model->unigrams[token]++;
        model->total_tokens++;
        if (i + 1 < seq->size) {
            bigram_add(&model->bigrams, seq->data[i], seq->data[i + 1], 1);
        }
        if (i + 2 < seq->size) {
            trigram_add(&model->trigrams, seq->data[i], seq->data[i + 1], seq->data[i + 2], 1);
        }
    }
}

static void train_model_from_corpus(Model *model, const char *corpus_path) {
    FILE *fp = fopen(corpus_path, "r");
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    IntVector seq;
    long long line_count = 0;
    if (!fp) {
        die("could not open corpus '%s': %s", corpus_path, strerror(errno));
    }

    model_add_special_tokens(model);
    int_vector_init(&seq);

    while ((line_len = getline(&line, &line_cap, fp)) != -1) {
        char *token;
        char *saveptr = NULL;
        char *token_ptrs[2048];
        int token_count = 0;
        (void) line_len;
        if (line[0] == '#'
            || is_blank_line(line)) {
            continue;
        }
        int_vector_clear(&seq);
        token = strtok_r(line, " \t\r\n", &saveptr);
        while (token) {
            int id = model_get_or_add_token(model, token);
            if (token_count >= ARRAY_LEN(token_ptrs)) {
                die("corpus line contains too many tokens");
            }
            token_ptrs[token_count++] = token;
            int_vector_push(&seq, id);
            token = strtok_r(NULL, " \t\r\n", &saveptr);
        }
        if (seq.size > 0) {
            model_add_examples_from_tokens(model, token_ptrs, token_count);
            model_count_sequence(model, &seq);
            line_count++;
        }
    }

    if (model->unk_id >= 0) {
        model->unigrams[model->unk_id] += 1;
        model->total_tokens += 1;
    }

    free(line);
    int_vector_free(&seq);
    fclose(fp);

    if (line_count == 0) {
        die("corpus '%s' did not contain any trainable lines", corpus_path);
    }
}

static void save_model(const Model *model, const char *model_path) {
    FILE *fp = fopen(model_path, "w");
    int i;
    if (!fp) {
        die("could not write model '%s': %s", model_path, strerror(errno));
    }

    fprintf(fp, "CHATBOTLM 1\n");
    fprintf(fp, "TOTAL %lld\n", model->total_tokens);
    fprintf(fp, "VOCAB %d\n", model->vocab.size);
    for (i = 0; i < model->vocab.size; i++) {
        fprintf(fp, "%s\t%d\n", model->vocab.tokens[i], model->unigrams[i]);
    }
    fprintf(fp, "BIGRAM %d\n", model->bigrams.size);
    for (i = 0; i < model->bigrams.cap; i++) {
        const BigramEntry *entry = &model->bigrams.entries[i];
        if (entry->used) {
            fprintf(fp, "%d %d %d\n", entry->a, entry->b, entry->count);
        }
    }
    fprintf(fp, "TRIGRAM %d\n", model->trigrams.size);
    for (i = 0; i < model->trigrams.cap; i++) {
        const TrigramEntry *entry = &model->trigrams.entries[i];
        if (entry->used) {
            fprintf(fp, "%d %d %d %d\n", entry->a, entry->b, entry->c, entry->count);
        }
    }
    fprintf(fp, "EXAMPLES %d\n", model->examples.size);
    for (i = 0; i < model->examples.size; i++) {
        fprintf(fp, "%s\t%s\n", model->examples.items[i].user_text, model->examples.items[i].assistant_text);
    }

    fclose(fp);
}

static void load_model(Model *model, const char *model_path) {
    FILE *fp = fopen(model_path, "r");
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    int expected_vocab = 0;
    int expected_bigrams = 0;
    int expected_trigrams = 0;
    int expected_examples = 0;
    int i;
    if (!fp) {
        die("could not open model '%s': %s", model_path, strerror(errno));
    }

    line_len = getline(&line, &line_cap, fp);
    if (line_len < 0 || strcmp(line, "CHATBOTLM 1\n") != 0) {
        die("model '%s' has an invalid header", model_path);
    }

    if (getline(&line, &line_cap, fp) < 0 || sscanf(line, "TOTAL %lld", &model->total_tokens) != 1) {
        die("model '%s' is missing total token metadata", model_path);
    }
    if (getline(&line, &line_cap, fp) < 0 || sscanf(line, "VOCAB %d", &expected_vocab) != 1) {
        die("model '%s' is missing vocabulary metadata", model_path);
    }

    for (i = 0; i < expected_vocab; i++) {
        char *tab;
        char *endptr;
        int id;
        int count;
        if (getline(&line, &line_cap, fp) < 0) {
            die("model '%s' ended unexpectedly while reading vocabulary", model_path);
        }
        tab = strchr(line, '\t');
        if (!tab) {
            die("model '%s' has a malformed vocabulary entry", model_path);
        }
        *tab = '\0';
        id = model_get_or_add_token(model, line);
        endptr = tab + 1;
        count = (int) strtol(endptr, NULL, 10);
        model->unigrams[id] = count;
    }

    if (getline(&line, &line_cap, fp) < 0 || sscanf(line, "BIGRAM %d", &expected_bigrams) != 1) {
        die("model '%s' is missing bigram metadata", model_path);
    }
    for (i = 0; i < expected_bigrams; i++) {
        int a;
        int b;
        int count;
        if (getline(&line, &line_cap, fp) < 0) {
            die("model '%s' ended unexpectedly while reading bigrams", model_path);
        }
        if (sscanf(line, "%d %d %d", &a, &b, &count) != 3) {
            die("model '%s' has a malformed bigram entry", model_path);
        }
        bigram_add(&model->bigrams, a, b, count);
    }

    if (getline(&line, &line_cap, fp) < 0 || sscanf(line, "TRIGRAM %d", &expected_trigrams) != 1) {
        die("model '%s' is missing trigram metadata", model_path);
    }
    for (i = 0; i < expected_trigrams; i++) {
        int a;
        int b;
        int c;
        int count;
        if (getline(&line, &line_cap, fp) < 0) {
            die("model '%s' ended unexpectedly while reading trigrams", model_path);
        }
        if (sscanf(line, "%d %d %d %d", &a, &b, &c, &count) != 4) {
            die("model '%s' has a malformed trigram entry", model_path);
        }
        trigram_add(&model->trigrams, a, b, c, count);
    }

    if (getline(&line, &line_cap, fp) < 0 || sscanf(line, "EXAMPLES %d", &expected_examples) != 1) {
        die("model '%s' is missing example metadata", model_path);
    }
    for (i = 0; i < expected_examples; i++) {
        char *tab;
        if (getline(&line, &line_cap, fp) < 0) {
            die("model '%s' ended unexpectedly while reading examples", model_path);
        }
        tab = strchr(line, '\t');
        if (!tab) {
            die("model '%s' has a malformed example entry", model_path);
        }
        *tab = '\0';
        {
            char *assistant = tab + 1;
            size_t len = strlen(assistant);
            while (len > 0 && (assistant[len - 1] == '\n' || assistant[len - 1] == '\r')) {
                assistant[--len] = '\0';
            }
            example_set_add(&model->examples, line, assistant);
        }
    }

    free(line);
    fclose(fp);
    model_resolve_special_ids(model);
}

static int is_punctuation_token(const char *token) {
    return strcmp(token, ".") == 0
        || strcmp(token, ",") == 0
        || strcmp(token, "!") == 0
        || strcmp(token, "?") == 0
        || strcmp(token, ":") == 0
        || strcmp(token, ";") == 0;
}

static int is_terminal_punctuation_token(const char *token) {
    return strcmp(token, ".") == 0
        || strcmp(token, "!") == 0
        || strcmp(token, "?") == 0;
}

static char *normalize_text(const char *input) {
    size_t len = strlen(input);
    char *output = xmalloc(len * 3 + 4);
    size_t j = 0;
    int last_was_space = 1;
    size_t i;

    for (i = 0; i < len; i++) {
        unsigned char ch = (unsigned char) input[i];
        if (isalnum(ch)) {
            output[j++] = (char) tolower(ch);
            last_was_space = 0;
        } else if (ch == '\'' ) {
            continue;
        } else if (strchr(".,!?;:()", ch)) {
            if (!last_was_space) {
                output[j++] = ' ';
            }
            output[j++] = (char) ch;
            output[j++] = ' ';
            last_was_space = 1;
        } else {
            if (!last_was_space) {
                output[j++] = ' ';
                last_was_space = 1;
            }
        }
    }

    while (j > 0 && output[j - 1] == ' ') {
        j--;
    }
    output[j] = '\0';
    return output;
}

static void append_prompt_tokens(const Model *model, const char *text, IntVector *seq) {
    char *normalized = normalize_text(text);
    char *saveptr = NULL;
    char *token = strtok_r(normalized, " ", &saveptr);
    while (token) {
        int id = vocab_lookup_id(&model->vocab, token);
        if (id < 0) {
            id = model->unk_id;
        }
        int_vector_push(seq, id);
        token = strtok_r(NULL, " ", &saveptr);
    }
    free(normalized);
}

static double token_probability(const Model *model, int a, int b, int c) {
    const double alpha = 0.10;
    double vocab_size = (double) model->vocab.size;
    double p1 = ((double) model->unigrams[c] + alpha)
        / ((double) model->total_tokens + alpha * vocab_size);
    double p2 = ((double) bigram_get(&model->bigrams, b, c) + alpha)
        / ((double) model->unigrams[b] + alpha * vocab_size);
    double p3 = ((double) trigram_get(&model->trigrams, a, b, c) + alpha)
        / ((double) bigram_get(&model->bigrams, a, b) + alpha * vocab_size);
    return 0.15 * p1 + 0.30 * p2 + 0.55 * p3;
}

static int is_stopword_token(const char *token) {
    static const char *stopwords[] = {
        "a", "an", "and", "are", "can", "do", "for", "i", "in", "is",
        "it", "me", "my", "of", "on", "or", "please", "the", "this",
        "to", "we", "with", "would", "you", "your"
    };
    int i;
    for (i = 0; i < ARRAY_LEN(stopwords); i++) {
        if (strcmp(token, stopwords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static double token_weight(const char *token) {
    if (strcmp(token, "c") == 0) {
        return 1.25;
    }
    if (is_stopword_token(token)) {
        return 0.35;
    }
    return 1.0;
}

static int whole_token_in_text(const char *text, const char *token) {
    size_t len = strlen(token);
    const char *cursor = text;
    if (len == 0) {
        return 0;
    }
    while ((cursor = strstr(cursor, token)) != NULL) {
        int left_ok = (cursor == text) || cursor[-1] == ' ';
        int right_ok = cursor[len] == '\0' || cursor[len] == ' ';
        if (left_ok && right_ok) {
            return 1;
        }
        cursor++;
    }
    return 0;
}

static double example_similarity(const char *prompt_norm, const char *example_norm) {
    char *prompt_copy = xstrdup(prompt_norm);
    char *example_copy = xstrdup(example_norm);
    char *token;
    char *saveptr = NULL;
    char *prompt_tokens[128];
    char *example_tokens[128];
    int prompt_count = 0;
    int example_count = 0;
    double prompt_shared = 0.0;
    double prompt_total = 0.0;
    double example_shared = 0.0;
    double example_total = 0.0;
    double bigram_hits = 0.0;
    double bigram_total = 0.0;
    int i;

    token = strtok_r(prompt_copy, " ", &saveptr);
    while (token && prompt_count < ARRAY_LEN(prompt_tokens)) {
        prompt_tokens[prompt_count++] = token;
        token = strtok_r(NULL, " ", &saveptr);
    }

    saveptr = NULL;
    token = strtok_r(example_copy, " ", &saveptr);
    while (token && example_count < ARRAY_LEN(example_tokens)) {
        example_tokens[example_count++] = token;
        token = strtok_r(NULL, " ", &saveptr);
    }

    for (i = 0; i < prompt_count; i++) {
        double weight;
        if (is_punctuation_token(prompt_tokens[i])) {
            continue;
        }
        weight = token_weight(prompt_tokens[i]);
        prompt_total += weight;
        if (whole_token_in_text(example_norm, prompt_tokens[i])) {
            prompt_shared += weight;
        }
    }

    for (i = 0; i < example_count; i++) {
        double weight;
        if (is_punctuation_token(example_tokens[i])) {
            continue;
        }
        weight = token_weight(example_tokens[i]);
        example_total += weight;
        if (whole_token_in_text(prompt_norm, example_tokens[i])) {
            example_shared += weight;
        }
    }

    for (i = 0; i + 1 < prompt_count; i++) {
        char phrase[128];
        double weight;
        if (is_punctuation_token(prompt_tokens[i]) || is_punctuation_token(prompt_tokens[i + 1])) {
            continue;
        }
        weight = (is_stopword_token(prompt_tokens[i]) || is_stopword_token(prompt_tokens[i + 1])) ? 0.20 : 0.85;
        snprintf(phrase, sizeof(phrase), "%s %s", prompt_tokens[i], prompt_tokens[i + 1]);
        bigram_total += weight;
        if (strstr(example_norm, phrase) != NULL) {
            bigram_hits += weight;
        }
    }

    free(prompt_copy);
    free(example_copy);

    if (prompt_total <= 0.0) {
        return 0.0;
    }
    {
        double recall = prompt_shared / prompt_total;
        double precision = example_total > 0.0 ? example_shared / example_total : 0.0;
        double phrase_score = bigram_total > 0.0 ? bigram_hits / bigram_total : 0.0;
        double score = 0.58 * recall + 0.32 * precision + 0.10 * phrase_score;
        if (prompt_norm[0] != '\0' && strstr(example_norm, prompt_norm) != NULL) {
            score += 0.08;
        }
        if (score > 1.0) {
            score = 1.0;
        }
        return score;
    }
}

static const DialogueExample *retrieve_best_example(const Model *model, const char *prompt, double *out_score) {
    char *prompt_norm = normalize_text(prompt);
    const DialogueExample *best = NULL;
    double best_score = 0.0;
    int i;

    for (i = 0; i < model->examples.size; i++) {
        double score = example_similarity(prompt_norm, model->examples.items[i].user_norm);
        if (score > best_score) {
            best_score = score;
            best = &model->examples.items[i];
        }
    }

    free(prompt_norm);
    if (out_score) {
        *out_score = best_score;
    }
    return best;
}

static void consider_candidate(Candidate *top, int *top_count, int id, double score) {
    int i;
    int insert_at = *top_count;
    if (*top_count == TOP_K && score <= top[*top_count - 1].score) {
        return;
    }
    if (*top_count < TOP_K) {
        (*top_count)++;
    }
    while (insert_at > 0 && top[insert_at - 1].score < score) {
        if (insert_at < TOP_K) {
            top[insert_at] = top[insert_at - 1];
        }
        insert_at--;
    }
    if (insert_at < TOP_K) {
        top[insert_at].id = id;
        top[insert_at].score = score;
    }
    for (i = *top_count; i < TOP_K; i++) {
        top[i].id = -1;
        top[i].score = 0.0;
    }
}

static int sample_next_token(const Model *model, int a, int b, int generated_count, double temperature) {
    Candidate top[TOP_K];
    int top_count = 0;
    int token_id;
    int i;
    double weights[TOP_K];
    double total_weight = 0.0;
    for (i = 0; i < TOP_K; i++) {
        top[i].id = -1;
        top[i].score = 0.0;
        weights[i] = 0.0;
    }

    for (token_id = 0; token_id < model->vocab.size; token_id++) {
        const char *token = model->vocab.tokens[token_id];
        double score;
        if (token_id == model->bos_id || token_id == model->user_id
            || token_id == model->assistant_id || token_id == model->unk_id) {
            continue;
        }

        score = token_probability(model, a, b, token_id);
        if (generated_count < 4 && token_id == model->end_id) {
            score *= 0.05;
        }
        if (generated_count > 0 && token_id == b && token_id != model->end_id) {
            score *= 0.20;
        }
        if (generated_count > 1 && token_id == a && token_id == b && token_id != model->end_id) {
            score *= 0.10;
        }
        if (generated_count >= 6 && token_id == model->end_id
            && is_terminal_punctuation_token(model->vocab.tokens[b])) {
            score *= 2.00;
        }
        if (is_punctuation_token(token) && is_punctuation_token(model->vocab.tokens[b])) {
            score *= 0.10;
        }
        consider_candidate(top, &top_count, token_id, score);
    }

    if (top_count == 0) {
        return model->end_id;
    }

    for (i = 0; i < top_count; i++) {
        double adjusted = pow(top[i].score + 1e-12, 1.0 / temperature);
        weights[i] = adjusted;
        total_weight += adjusted;
    }

    if (total_weight <= 0.0) {
        return top[0].id;
    }

    {
        double r = ((double) rand() / (double) RAND_MAX) * total_weight;
        double running = 0.0;
        for (i = 0; i < top_count; i++) {
            running += weights[i];
            if (r <= running) {
                return top[i].id;
            }
        }
    }

    return top[top_count - 1].id;
}

static char *detokenize_reply(const Model *model, const IntVector *reply_tokens) {
    size_t cap = 256;
    size_t len = 0;
    char *output = xmalloc(cap);
    int i;
    int capitalize_next = 1;

    output[0] = '\0';
    for (i = 0; i < reply_tokens->size; i++) {
        const char *token = model->vocab.tokens[reply_tokens->data[i]];
        size_t token_len = strlen(token);
        if (len + token_len + 3 >= cap) {
            while (len + token_len + 3 >= cap) {
                cap *= 2;
            }
            output = xrealloc(output, cap);
        }
        if (is_punctuation_token(token)) {
            if (len > 0 && output[len - 1] == ' ') {
                len--;
            }
            memcpy(output + len, token, token_len);
            len += token_len;
            output[len++] = ' ';
            output[len] = '\0';
            if (is_terminal_punctuation_token(token)) {
                capitalize_next = 1;
            }
            continue;
        }
        if (len > 0 && output[len - 1] != ' ') {
            output[len++] = ' ';
        }
        memcpy(output + len, token, token_len);
        if (capitalize_next) {
            output[len] = (char) toupper((unsigned char) output[len]);
            capitalize_next = 0;
        }
        len += token_len;
        output[len++] = ' ';
        output[len] = '\0';
    }

    while (len > 0 && output[len - 1] == ' ') {
        len--;
    }
    output[len] = '\0';

    if (len == 0) {
        free(output);
        return xstrdup("I can help with that.");
    }
    return output;
}

static char *generate_reply(const Model *model, const char *prompt, int max_reply_tokens, double temperature) {
    const DialogueExample *best_example;
    double best_score = 0.0;
    IntVector context;
    IntVector reply;
    int a;
    int b;
    int i;
    char *text;

    best_example = retrieve_best_example(model, prompt, &best_score);
    if (best_example && best_score >= 0.30) {
        return xstrdup(best_example->assistant_text);
    }

    int_vector_init(&context);
    int_vector_init(&reply);

    int_vector_push(&context, model->bos_id);
    int_vector_push(&context, model->bos_id);
    int_vector_push(&context, model->user_id);
    append_prompt_tokens(model, prompt, &context);
    int_vector_push(&context, model->assistant_id);

    if (context.size < 2) {
        die("generation context is too short");
    }

    for (i = 0; i < max_reply_tokens; i++) {
        int next;
        a = context.data[context.size - 2];
        b = context.data[context.size - 1];
        next = sample_next_token(model, a, b, reply.size, temperature);
        if (next == model->end_id || next == model->user_id || next == model->assistant_id) {
            break;
        }
        int_vector_push(&context, next);
        int_vector_push(&reply, next);
    }

    text = detokenize_reply(model, &reply);
    int_vector_free(&context);
    int_vector_free(&reply);
    return text;
}

static long write_dialogue(FILE *fp, const char *user, const char *assistant) {
    if (fprintf(fp, "<bos> <bos> <user> %s <assistant> %s <end>\n", user, assistant) < 0) {
        die("failed to write generated dialogue");
    }
    return 1;
}

static long write_two_turn_dialogue(FILE *fp,
                                    const char *user1,
                                    const char *assistant1,
                                    const char *user2,
                                    const char *assistant2) {
    if (fprintf(fp,
                "<bos> <bos> <user> %s <assistant> %s <user> %s <assistant> %s <end>\n",
                user1,
                assistant1,
                user2,
                assistant2) < 0) {
        die("failed to write generated two turn dialogue");
    }
    return 1;
}

static long generate_dataset(const char *output_path) {
    static const char *greeting_users[] = {
        "hello .",
        "hi .",
        "hey .",
        "hello there .",
        "hi there .",
        "good morning .",
        "good afternoon .",
        "good evening .",
        "greetings .",
        "howdy .",
        "nice to meet you .",
        "hey chatbot .",
        "hello bot .",
        "good day .",
        "yo .",
        "what is up ?",
        "hi assistant .",
        "hey there ."
    };
    static const char *greeting_replies[] = {
        "hello . how can i help you today ?",
        "hi . what would you like to work on ?",
        "hey . i am ready to help .",
        "hello there . what do you need ?",
        "hi . ask me anything practical .",
        "good to hear from you . how can i assist ?",
        "hello . i can help with coding writing planning and study tasks .",
        "hi there . what should we solve first ?",
        "hey . tell me what you need and i will keep it clear .",
        "hello . i am here and ready to help .",
        "good day . what can i do for you ?",
        "hi . we can work through it step by step .",
        "hello . what problem are you trying to solve ?",
        "hey there . what would you like me to explain ?"
    };
    static const char *identity_users[] = {
        "who are you ?",
        "what are you ?",
        "tell me about yourself .",
        "what is your job ?",
        "what kind of bot are you ?",
        "are you a chatbot ?",
        "what do you do ?",
        "what should i call you ?",
        "how would you describe yourself ?",
        "what are your strengths ?",
        "are you useful for coding ?",
        "can you explain what you are ?"
    };
    static const char *identity_traits[] = {
        "clear",
        "practical",
        "focused",
        "calm",
        "helpful",
        "direct",
        "patient",
        "organized",
        "technical",
        "careful",
        "concise",
        "steady"
    };
    static const char *identity_endings[] = {
        "i work best on short practical requests .",
        "i try to give concrete next steps .",
        "i aim to stay useful and easy to follow .",
        "i am built to respond like a small assistant ."
    };
    static const char *capability_domains[] = {
        "coding",
        "debugging",
        "writing",
        "summaries",
        "study plans",
        "math explanations",
        "project planning",
        "research notes",
        "translation practice",
        "technical walkthroughs"
    };
    static const char *capability_user_forms[] = {
        "can you help with %s ?",
        "are you good at %s ?",
        "could you assist with %s ?",
        "can i use you for %s ?",
        "would you help me with %s ?",
        "can you support %s ?"
    };
    static const char *capability_replies[] = {
        "yes . i can help with %s and keep the answer practical .",
        "i can help with %s . give me the goal and any constraints .",
        "yes . i can work on %s step by step .",
        "i can assist with %s . a clear prompt usually works best ."
    };
    static const char *coding_tasks[] = {
        "write a c function that reverses a string .",
        "write a c program that counts words in a file .",
        "build a linked list implementation in c .",
        "explain pointers in c .",
        "show how structs work in c .",
        "help me parse csv data in c .",
        "write a small http request parser in c .",
        "create a stack implementation in c .",
        "create a queue implementation in c .",
        "write a binary search function in c .",
        "show a merge sort in c .",
        "write a tokenizer in c .",
        "explain memory allocation in c .",
        "write a simple command line parser in c .",
        "help me read json like text in c .",
        "write a hash table in c .",
        "explain file io in c .",
        "write unit tests for a c function .",
        "review a c function for bugs .",
        "optimize a loop in c .",
        "explain recursion with a c example .",
        "write a matrix multiply function in c .",
        "show how to split strings in c .",
        "build a tiny cache in c ."
    };
    static const char *coding_constraints[] = {
        "keep it simple .",
        "focus on readability .",
        "include comments .",
        "mention edge cases .",
        "avoid unnecessary complexity .",
        "show the main idea first .",
        "make it beginner friendly .",
        "keep the explanation short ."
    };
    static const char *coding_assistant_styles[] = {
        "yes . i can help with that and keep it structured .",
        "sure . i can handle that and explain the moving parts .",
        "i can do that . i will stay practical and clear .",
        "yes . i can work through that step by step .",
        "i can help . i will focus on the important details first ."
    };
    static const char *debug_issues[] = {
        "segfaults when i dereference a pointer",
        "prints garbage values",
        "crashes on large input",
        "leaks memory",
        "hangs in an infinite loop",
        "returns the wrong result",
        "fails to open a file",
        "parses only part of the input",
        "writes past the end of an array",
        "double frees memory",
        "misuses a string buffer",
        "fails when there are spaces in the input",
        "sorts values in the wrong order",
        "does not handle empty input",
        "stops after the first line",
        "breaks when i use unicode text"
    };
    static const char *debug_checks[] = {
        "checking pointer lifetimes",
        "verifying array bounds",
        "printing intermediate values",
        "checking return codes",
        "testing a minimal input",
        "reviewing loop conditions",
        "validating buffer sizes",
        "logging parsed tokens",
        "testing the empty case",
        "isolating one function at a time"
    };
    static const char *debug_followups[] = {
        "then compare the observed behavior with the expected result .",
        "then tighten the failing case into a small reproducible example .",
        "then fix one issue at a time and retest .",
        "then rerun the smallest test that shows the bug ."
    };
    static const char *writing_tasks[] = {
        "rewrite an email",
        "draft a project update",
        "summarize meeting notes",
        "improve a short bio",
        "edit a support reply",
        "write a polite follow up",
        "turn rough notes into a paragraph",
        "write a release note",
        "improve a bug report",
        "write a commit message",
        "tighten a proposal",
        "rewrite a paragraph",
        "write a short cover letter",
        "improve an introduction",
        "draft a user guide section"
    };
    static const char *writing_tones[] = {
        "formal",
        "friendly",
        "clear",
        "professional",
        "warm",
        "direct",
        "confident",
        "concise",
        "calm",
        "persuasive"
    };
    static const char *writing_lengths[] = {
        "keep it short .",
        "make it compact .",
        "use a medium length response .",
        "make it detailed but easy to scan ."
    };
    static const char *study_subjects[] = {
        "c programming",
        "data structures",
        "algorithms",
        "computer networks",
        "linux basics",
        "web development",
        "database design",
        "statistics",
        "linear algebra",
        "writing skills",
        "technical reading",
        "machine learning basics",
        "cybersecurity fundamentals",
        "operating systems"
    };
    static const char *study_goals[] = {
        "for an exam",
        "for an interview",
        "for a personal project",
        "for a weekly study routine",
        "for a beginner roadmap",
        "for steady daily progress",
        "for a revision plan",
        "for a hands on approach",
        "for a one month sprint",
        "for better understanding"
    };
    static const char *study_plan_shapes[] = {
        "i can break it into clear milestones and daily tasks .",
        "i can turn that into a practical study plan with checkpoints .",
        "i can organize that into small lessons and review sessions .",
        "i can build a plan that balances learning and practice ."
    };
    static const char *summary_sources[] = {
        "an article",
        "a chapter",
        "a tutorial",
        "a bug report",
        "meeting notes",
        "research notes",
        "a design document",
        "an email thread",
        "a user manual",
        "a case study",
        "a pull request",
        "an interview transcript",
        "a textbook section",
        "a product brief"
    };
    static const char *summary_audiences[] = {
        "for a beginner",
        "for a busy manager",
        "for a developer",
        "for a student",
        "for quick review",
        "for a handoff note",
        "for a teammate",
        "for presentation prep"
    };
    static const char *summary_depths[] = {
        "i can keep the key points only .",
        "i can keep the summary short and concrete .",
        "i can include the main idea and the most important details .",
        "i can summarize it in plain language and highlight action items ."
    };
    static const char *math_tasks[] = {
        "solve a linear equation",
        "explain fractions",
        "show basic probability",
        "walk through percentages",
        "explain averages",
        "work through exponents",
        "explain prime numbers",
        "show how ratios work",
        "explain standard deviation",
        "walk through matrix multiplication",
        "explain gradients",
        "show a simple derivative",
        "explain bayes rule",
        "walk through binary numbers",
        "explain logarithms",
        "show how modulo works"
    };
    static const char *math_styles[] = {
        "step by step",
        "with a small example",
        "using plain language",
        "with a visual intuition",
        "for a beginner",
        "with the key formula first",
        "in a short practical way",
        "with a quick check at the end"
    };
    static const char *math_replies[] = {
        "i can explain that clearly and keep each step visible .",
        "yes . i can walk through it and show a simple example .",
        "i can help . i will keep the math concrete and easy to follow .",
        "sure . i can break it down and check the result at the end ."
    };
    static const char *support_situations[] = {
        "i feel stuck on a task .",
        "i am overwhelmed by my to do list .",
        "i do not know where to start .",
        "my project feels too large .",
        "i keep procrastinating .",
        "i am worried about an exam .",
        "i have too many tabs open .",
        "i need help focusing .",
        "i am behind on a deadline .",
        "my notes are a mess .",
        "i keep changing direction .",
        "i have been debugging for hours .",
        "i need a calm plan .",
        "i am not sure which task matters most ."
    };
    static const char *support_actions[] = {
        "start with the smallest useful step",
        "pick one clear priority",
        "turn the work into a short checklist",
        "set a ten minute starting task",
        "define what done looks like",
        "separate urgent work from optional work",
        "write down the next concrete action",
        "reduce the task until it feels manageable"
    };
    static const char *support_pacing[] = {
        "and keep the plan calm and realistic .",
        "and make the next move easy to start .",
        "and avoid adding unnecessary pressure .",
        "and keep the first step very small ."
    };
    static const char *translation_phrases[] = {
        "translate hello how are you",
        "translate thank you for your help",
        "translate where is the station",
        "translate i need more time",
        "translate this file is missing",
        "translate please send the update",
        "translate can we meet tomorrow",
        "translate i understand the main idea",
        "translate the server is down",
        "translate keep the answer simple",
        "translate i will review the code",
        "translate the meeting starts at noon"
    };
    static const char *translation_languages[] = {
        "to swahili",
        "to french",
        "to spanish",
        "to german",
        "to arabic",
        "to portuguese",
        "to hindi",
        "to japanese"
    };
    static const char *translation_details[] = {
        "i can translate it and keep the meaning natural .",
        "i can translate it and explain any tricky phrase .",
        "i can give a simple translation first and then refine it .",
        "i can translate it and keep the wording clear ."
    };
    static const char *followup_topics[] = {
        "recursion",
        "pointers",
        "linked lists",
        "file io",
        "csv parsing",
        "sorting",
        "hash tables",
        "structs",
        "loops",
        "memory allocation"
    };
    static const char *followup_prompts[] = {
        "show a tiny c example .",
        "keep the explanation even simpler .",
        "what is the main pitfall ?",
        "how should i practice this ?"
    };
    static const char *followup_answers[] = {
        "here is a small c style example and the key idea behind it .",
        "i can simplify it further and focus on the one concept that matters most .",
        "the main pitfall is losing track of state and assumptions .",
        "practice with a tiny example first and then add one feature at a time ."
    };

    FILE *fp = fopen(output_path, "w");
    long count = 0;
    int i;
    int j;
    int k;

    if (!fp) {
        die("could not create dataset '%s': %s", output_path, strerror(errno));
    }

    for (i = 0; i < ARRAY_LEN(greeting_users); i++) {
        for (j = 0; j < ARRAY_LEN(greeting_replies); j++) {
            count += write_dialogue(fp, greeting_users[i], greeting_replies[j]);
        }
    }

    for (i = 0; i < ARRAY_LEN(identity_users); i++) {
        for (j = 0; j < ARRAY_LEN(identity_traits); j++) {
            for (k = 0; k < ARRAY_LEN(identity_endings); k++) {
                char assistant[512];
                snprintf(assistant,
                         sizeof(assistant),
                         "i am a small chatbot trained from a dialogue corpus . i try to be %s . %s",
                         identity_traits[j],
                         identity_endings[k]);
                count += write_dialogue(fp, identity_users[i], assistant);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(capability_domains); i++) {
        for (j = 0; j < ARRAY_LEN(capability_user_forms); j++) {
            for (k = 0; k < ARRAY_LEN(capability_replies); k++) {
                char user[256];
                char assistant[256];
                snprintf(user, sizeof(user), capability_user_forms[j], capability_domains[i]);
                snprintf(assistant, sizeof(assistant), capability_replies[k], capability_domains[i]);
                count += write_dialogue(fp, user, assistant);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(coding_tasks); i++) {
        for (j = 0; j < ARRAY_LEN(coding_constraints); j++) {
            for (k = 0; k < ARRAY_LEN(coding_assistant_styles); k++) {
                char user[512];
                char assistant[1024];
                snprintf(user, sizeof(user), "%s %s", coding_tasks[i], coding_constraints[j]);
                snprintf(assistant,
                         sizeof(assistant),
                         "%s i will help with this request . i will keep the solution practical and mention the core steps . %s",
                         coding_assistant_styles[k],
                         coding_constraints[j]);
                count += write_dialogue(fp, user, assistant);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(debug_issues); i++) {
        for (j = 0; j < ARRAY_LEN(debug_checks); j++) {
            for (k = 0; k < ARRAY_LEN(debug_followups); k++) {
                char user[512];
                char assistant[1024];
                snprintf(user, sizeof(user), "my c program %s .", debug_issues[i]);
                snprintf(assistant,
                         sizeof(assistant),
                         "start by %s . %s",
                         debug_checks[j],
                         debug_followups[k]);
                count += write_dialogue(fp, user, assistant);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(writing_tasks); i++) {
        for (j = 0; j < ARRAY_LEN(writing_tones); j++) {
            for (k = 0; k < ARRAY_LEN(writing_lengths); k++) {
                char user[512];
                char assistant[1024];
                snprintf(user, sizeof(user), "help me %s in a %s tone . %s", writing_tasks[i], writing_tones[j], writing_lengths[k]);
                snprintf(assistant,
                         sizeof(assistant),
                         "i can do that . i will keep it %s and easy to read . %s",
                         writing_tones[j],
                         writing_lengths[k]);
                count += write_dialogue(fp, user, assistant);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(study_subjects); i++) {
        for (j = 0; j < ARRAY_LEN(study_goals); j++) {
            for (k = 0; k < ARRAY_LEN(study_plan_shapes); k++) {
                char user[512];
                snprintf(user, sizeof(user), "make a study plan for %s %s .", study_subjects[i], study_goals[j]);
                count += write_dialogue(fp, user, study_plan_shapes[k]);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(summary_sources); i++) {
        for (j = 0; j < ARRAY_LEN(summary_audiences); j++) {
            for (k = 0; k < ARRAY_LEN(summary_depths); k++) {
                char user[512];
                snprintf(user, sizeof(user), "summarize %s %s .", summary_sources[i], summary_audiences[j]);
                count += write_dialogue(fp, user, summary_depths[k]);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(math_tasks); i++) {
        for (j = 0; j < ARRAY_LEN(math_styles); j++) {
            for (k = 0; k < ARRAY_LEN(math_replies); k++) {
                char user[512];
                char assistant[1024];
                snprintf(user, sizeof(user), "%s %s .", math_tasks[i], math_styles[j]);
                snprintf(assistant, sizeof(assistant), "%s i will explain it %s .", math_replies[k], math_styles[j]);
                count += write_dialogue(fp, user, assistant);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(support_situations); i++) {
        for (j = 0; j < ARRAY_LEN(support_actions); j++) {
            for (k = 0; k < ARRAY_LEN(support_pacing); k++) {
                char assistant[1024];
                snprintf(assistant, sizeof(assistant), "let us %s %s", support_actions[j], support_pacing[k]);
                count += write_dialogue(fp, support_situations[i], assistant);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(translation_phrases); i++) {
        for (j = 0; j < ARRAY_LEN(translation_languages); j++) {
            for (k = 0; k < ARRAY_LEN(translation_details); k++) {
                char user[512];
                snprintf(user, sizeof(user), "%s %s .", translation_phrases[i], translation_languages[j]);
                count += write_dialogue(fp, user, translation_details[k]);
            }
        }
    }

    for (i = 0; i < ARRAY_LEN(followup_topics); i++) {
        for (j = 0; j < ARRAY_LEN(followup_prompts); j++) {
            count += write_two_turn_dialogue(
                fp,
                "explain this topic in c .",
                "sure . tell me the topic and i will break it down .",
                followup_prompts[j],
                followup_answers[j]
            );
            {
                char user1[256];
                char assistant1[512];
                snprintf(user1, sizeof(user1), "explain %s in simple terms .", followup_topics[i]);
                snprintf(assistant1,
                         sizeof(assistant1),
                         "%s is easier to learn when you focus on one example at a time .",
                         followup_topics[i]);
                count += write_two_turn_dialogue(fp, user1, assistant1, followup_prompts[j], followup_answers[j]);
            }
        }
    }

    fclose(fp);
    return count;
}

static void print_usage(FILE *stream) {
    fprintf(stream,
            "usage:\n"
            "  chatbot_lm generate-data <output-corpus>\n"
            "  chatbot_lm train <corpus> <model>\n"
            "  chatbot_lm reply <model> <prompt...>\n"
            "  chatbot_lm chat <model>\n");
}

static char *join_arguments(int argc, char **argv, int start) {
    int i;
    size_t total = 0;
    char *result;
    size_t cursor = 0;
    for (i = start; i < argc; i++) {
        total += strlen(argv[i]) + 1;
    }
    result = xmalloc(total + 1);
    result[0] = '\0';
    for (i = start; i < argc; i++) {
        size_t len = strlen(argv[i]);
        memcpy(result + cursor, argv[i], len);
        cursor += len;
        if (i + 1 < argc) {
            result[cursor++] = ' ';
        }
    }
    result[cursor] = '\0';
    return result;
}

static void run_chat_repl(const Model *model) {
    char *line = NULL;
    size_t cap = 0;
    printf("type 'exit' or 'quit' to stop.\n");
    while (1) {
        char *reply;
        printf("you> ");
        fflush(stdout);
        if (getline(&line, &cap, stdin) < 0) {
            break;
        }
        {
            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                line[--len] = '\0';
            }
        }
        if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) {
            break;
        }
        if (is_blank_line(line)) {
            continue;
        }
        reply = generate_reply(model, line, DEFAULT_MAX_REPLY_TOKENS, DEFAULT_TEMPERATURE);
        printf("bot> %s\n", reply);
        free(reply);
    }
    free(line);
}

int main(int argc, char **argv) {
    srand((unsigned int) time(NULL));

    if (argc < 2) {
        print_usage(stderr);
        return 1;
    }

    if (strcmp(argv[1], "generate-data") == 0) {
        long rows;
        if (argc != 3) {
            print_usage(stderr);
            return 1;
        }
        rows = generate_dataset(argv[2]);
        printf("generated %ld chat dialogues in %s\n", rows, argv[2]);
        return 0;
    }

    if (strcmp(argv[1], "train") == 0) {
        Model model;
        if (argc != 4) {
            print_usage(stderr);
            return 1;
        }
        model_init(&model);
        train_model_from_corpus(&model, argv[2]);
        save_model(&model, argv[3]);
        printf("trained model: vocab=%d total_tokens=%lld bigrams=%d trigrams=%d examples=%d saved_to=%s\n",
               model.vocab.size,
               model.total_tokens,
               model.bigrams.size,
               model.trigrams.size,
               model.examples.size,
               argv[3]);
        model_free(&model);
        return 0;
    }

    if (strcmp(argv[1], "reply") == 0) {
        Model model;
        char *prompt;
        char *reply;
        if (argc < 4) {
            print_usage(stderr);
            return 1;
        }
        prompt = join_arguments(argc, argv, 3);
        model_init(&model);
        load_model(&model, argv[2]);
        reply = generate_reply(&model, prompt, DEFAULT_MAX_REPLY_TOKENS, DEFAULT_TEMPERATURE);
        printf("%s\n", reply);
        free(reply);
        free(prompt);
        model_free(&model);
        return 0;
    }

    if (strcmp(argv[1], "chat") == 0) {
        Model model;
        if (argc != 3) {
            print_usage(stderr);
            return 1;
        }
        model_init(&model);
        load_model(&model, argv[2]);
        run_chat_repl(&model);
        model_free(&model);
        return 0;
    }

    print_usage(stderr);
    return 1;
}
