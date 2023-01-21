#ifndef _RING_BUFFER_H_
#define _RING_BUFFER_H_

#include <stdint.h>
#ifndef TEST
#include "model.h"
#include "PREDMNT1_config.h"
#endif

#ifdef TEST
#define RING_BUFFER_SIZE 20
#else
#define RING_BUFFER_SIZE 600
#endif
#define RING_BUFFER_ELEMENT_SIZE 6

#ifdef TEST
#define ai_float int32_t
#define _PRINTF printf
#endif

int32_t ring_buffer_get_index();


void ring_buffer_store_data(int32_t *data);
// read the last N elements from the buffer
void ring_buffer_read_data(ai_float *data, int32_t N);
void ring_buffer_init();
void ring_buffer_print_buffer();
void ring_buffer_get_min(ai_float* min_acc, ai_float* min_gyro);
void ring_buffer_get_max(ai_float* max_acc, ai_float* max_gyro);

#endif // _RING_BUFFER_H_
