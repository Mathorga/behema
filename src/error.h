#ifndef __PORTIA_ERROR__
#define __PORTIA_ERROR__

typedef enum error_code_t {
    ERROR_NONE = 0,
    ERROR_NH_RADIUS_TOO_BIG = 1,
    ERROR_FILE_DOES_NOT_EXIST = 2,
    ERROR_FILE_SIZE_WRONG = 3,
    ERROR_FAILED_ALLOC = 4
} error_code_t;

#endif