#ifndef __behema_ERROR__
#define __behema_ERROR__

typedef enum {
    ERROR_NONE = 0,
    ERROR_NH_RADIUS_TOO_BIG = 1,
    ERROR_FILE_DOES_NOT_EXIST = 2,
    ERROR_FILE_SIZE_WRONG = 3,
    ERROR_FAILED_ALLOC = 4,
    ERROR_CORTEX_UNALLOC = 5,
    ERROR_SIZE_WRONG = 6
} error_code_t;

#endif
