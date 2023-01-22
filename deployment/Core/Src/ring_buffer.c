// define a circular buffer for storing data from the ADC , and a function to read the data from the buffer . each element is an 6 byte array

#include "ring_buffer.h"

// define the buffer
static int32_t buffer[RING_BUFFER_SIZE][RING_BUFFER_ELEMENT_SIZE];
static int32_t buffer_index = 0;

static ai_float min_acc = INT32_MAX;
static ai_float max_acc = INT32_MIN;

static ai_float min_gyro = INT32_MAX;
static ai_float max_gyro = INT32_MIN;


static float current_velocity[3] = {0.0f, 0.0f, 0.0f};
static float current_gravity[3] = {0.0f, 0.0f, 0.0f};

static float VectorMagnitude(const float* vec);

static float VectorMagnitude(const float* vec) 
{
  const float x = vec[0];
  const float y = vec[1];
  const float z = vec[2];
  return sqrtf((x * x) + (y * y) + (z * z));
}




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
#ifdef TEST
void ring_buffer_store_data(float *data)
#else
void ring_buffer_store_data(int32_t *data)
#endif
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

    min_acc = INT32_MAX;
    max_acc = INT32_MIN;
    min_gyro = INT32_MAX;
    max_gyro = INT32_MIN;

    for (j = 0; j < RING_BUFFER_ELEMENT_SIZE/2; j++)
    {
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


void ring_buffer_estimate_velocity(int new_samples)
{
    const float gravity_x = current_gravity[0];
    const float gravity_y = current_gravity[1];
    const float gravity_z = current_gravity[2];

    const float friction_fudge = 0.98f;

    int32_t i;
    int32_t index = buffer_index;
    int32_t count = new_samples;

    if (new_samples > RING_BUFFER_SIZE)
    {
        count = RING_BUFFER_SIZE;
    }

    for (i = 0; i < count; i++)
    {
        index = (index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;

        const int32_t* entry = &buffer[index][0];
        const float ax = (float)entry[0];
        const float ay = (float)entry[1];
        const float az = (float)entry[2];

        // Try to remove gravity from the raw acceleration values.
        const float ax_minus_gravity = ax - gravity_x;
        const float ay_minus_gravity = ay - gravity_y;
        const float az_minus_gravity = az - gravity_z;

        // Update velocity based on the normalized acceleration.
        current_velocity[0] += ax_minus_gravity;
        current_velocity[1] += ay_minus_gravity;
        current_velocity[2] += az_minus_gravity;

        // Dampen the velocity slightly with a fudge factor to stop it exploding.
        current_velocity[0] *= friction_fudge;
        current_velocity[1] *= friction_fudge;
        current_velocity[2] *= friction_fudge;

    }

    int32_t left_over = new_samples - count;
    for (i = 0; i < left_over; i++)
    {

//         const float * entry =   &buffer[RING_BUFFER_SIZE - left_over + i][0];
    }
}
