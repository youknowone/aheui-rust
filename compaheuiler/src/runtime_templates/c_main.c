
int64_t compaheuiler_c_entry(void) {
    int64_t* data[STORAGE_COUNT];
    int64_t* bases[STORAGE_COUNT];
    int32_t lengths[STORAGE_COUNT];
    for (int i = 0; i < STORAGE_COUNT; i++) {
        data[i] = (int64_t*)calloc(MAX_STACK, sizeof(int64_t));
        bases[i] = data[i];
        lengths[i] = 0;
    }
    int64_t result = aheui_main(bases, lengths);
    _flush();
    fflush(stdout);
    for (int i = 0; i < STORAGE_COUNT; i++) free(data[i]);
    return result;
}

#ifndef COMPAHEUILER_RUST_BIGINT
int main(void) { return (int)compaheuiler_c_entry(); }
#endif
