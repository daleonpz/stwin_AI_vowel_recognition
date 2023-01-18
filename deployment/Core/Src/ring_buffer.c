// define a circular buffer for storing data from the ADC , and a function to read the data from the buffer . each element is an 6 byte array

#include "ring_buffer.h"

// define the buffer
static int32_t buffer[RING_BUFFER_SIZE][RING_BUFFER_ELEMENT_SIZE];
static int32_t buffer_index = 0;

static ai_float min_acc = INT32_MAX;
static ai_float max_acc = INT32_MIN;

static ai_float min_gyro = INT32_MAX;
static ai_float max_gyro = INT32_MIN;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// initialize the buffer
void ring_buffer_init()
{
    for(int i = 0; i < RING_BUFFER_SIZE; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            buffer[i][j] = 0;
        }
    }
    buffer_index = 0;
}

// get index
int32_t ring_buffer_get_index()
{
    return buffer_index;
}

// store data in the buffer
void ring_buffer_store_data(int32_t *data) 
{
    int32_t i;
    for (i = 0; i < RING_BUFFER_ELEMENT_SIZE; i++) {
        buffer[buffer_index][i] = data[i];
    }
    buffer_index = (buffer_index + 1) % RING_BUFFER_SIZE;
}

// read the last N elements from the buffer
void ring_buffer_read_data( ai_float* data, int32_t N)
{
    int32_t i;
    int32_t j;
    int32_t index = buffer_index;
    int32_t count = N;

    ai_float min_array[6] = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
    ai_float max_array[6] = {INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};

    if (N > RING_BUFFER_SIZE)
    {
        count = RING_BUFFER_SIZE;
    }

    for (i = 0; i < count; i++)
    {
        index = (index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
        for (j = 0; j < RING_BUFFER_ELEMENT_SIZE; j++)
        {
            int32_t idx = (N - i - 1) * RING_BUFFER_ELEMENT_SIZE + j;
            data[idx] = (ai_float)buffer[index][j];
            min_array[j] = MIN(min_array[j], data[idx]);
            max_array[j] = MAX(max_array[j], data[idx]);

//             data[i * RING_BUFFER_ELEMENT_SIZE + j] = (ai_float)buffer[index][j] ; 
//             min_array[j] = MIN(min_array[j], data[i * RING_BUFFER_ELEMENT_SIZE + j]);
//             max_array[j] = MAX(max_array[j], data[i * RING_BUFFER_ELEMENT_SIZE + j]);
        }
    }

    int32_t left_over = N - count;
    for (i = 0; i < left_over; i++)
    {
        for (j = 0; j < RING_BUFFER_ELEMENT_SIZE; j++)
        {
            int32_t idx = (N - 1 - count - i) * RING_BUFFER_ELEMENT_SIZE + j;
            data[idx] = (ai_float)buffer[RING_BUFFER_SIZE - left_over + i][j];
            min_array[j] = MIN(min_array[j], data[idx]);
            max_array[j] = MAX(max_array[j], data[idx]);
        }
    }

//     for (i = count; i < N; i++)
//     {
//         for (j = 0; j < RING_BUFFER_ELEMENT_SIZE; j++)
//         {
//             data[i * RING_BUFFER_ELEMENT_SIZE + j] = (ai_float)buffer[RING_BUFFER_SIZE - (N-count) + i][j] ; 
//             min_array[j] = MIN(min_array[j], data[i * RING_BUFFER_ELEMENT_SIZE + j]);
//             max_array[j] = MAX(max_array[j], data[i * RING_BUFFER_ELEMENT_SIZE + j]);
//         }
//     }

//     for (j = 0; j < RING_BUFFER_ELEMENT_SIZE; j++)
//     {
//         _PRINTF("min_array[%d] = %d max_array[%d] = %d \r\n", j, min_array[j], j, max_array[j]);
//     }

    for (j = 0; j < RING_BUFFER_ELEMENT_SIZE/2; j++)
    {
//         _PRINTF("min_acc = %d max_acc = %d \r\n", min_acc, max_acc);
//         _PRINTF("min_gyro = %d max_gyro = %d \r\n", min_gyro, max_gyro);
//         _PRINTF("min_array[%d] = %d max_array[%d] = %d \r\n", j, min_array[j], j, max_array[j]);
//         _PRINTF("min_array[%d] = %d max_array[%d] = %d \r\n", j+3, min_array[j+3], j+3, max_array[j+3]);
        min_acc = MIN(min_acc, min_array[j]);
        max_acc = MAX(max_acc, max_array[j]);

        min_gyro = MIN(min_gyro, min_array[j+3]);
        max_gyro = MAX(max_gyro, max_array[j+3]);
    }

}

void ring_buffer_print_buffer()
{
    for(int i = 0; i < RING_BUFFER_SIZE; i++)
    {
        for(int j = 0; j < RING_BUFFER_ELEMENT_SIZE; j++)
        {
            _PRINTF("%ld ", buffer[i][j]);
        }
        _PRINTF("\r\n");
    }
}

void ring_buffer_get_min(ai_float* _min_acc, ai_float* _min_gyro)
{
    *_min_acc = min_acc;
    *_min_gyro = min_gyro;
}

void ring_buffer_get_max(ai_float* _max_acc, ai_float* _max_gyro)
{
    *_max_acc = max_acc;
    *_max_gyro = max_gyro;
}
