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


void ring_buffer_estimate_gravity(int32_t new_samples) 
{
    int32_t samples_to_average = MIN(new_samples, 100);
    int32_t size = RING_BUFFER_ELEMENT_SIZE * samples_to_average;
    ai_float new_data[size] ;
    ring_buffer_read_data(new_data, samples_to_average);

    current_gravity[0] = 0.0f;
    current_gravity[1] = 0.0f;
    current_gravity[2] = 0.0f;

    float x_total = 0.0f;
    float y_total = 0.0f;
    float z_total = 0.0f;

    for (int i = 0; i < samples_to_average; ++i) {
        const ai_float* entry = &new_data[i * RING_BUFFER_ELEMENT_SIZE];
        const float x = (float)entry[0];
        const float y = (float)entry[1];
        const float z = (float)entry[2];
        x_total += x;
        y_total += y;
        z_total += z;
    }
    current_gravity[0] = x_total / samples_to_average;
    current_gravity[1] = y_total / samples_to_average;
    current_gravity[2] = z_total / samples_to_average;
}

void ring_buffer_estimate_velocity(int32_t new_samples, float dt)
{
    int32_t size = new_samples * RING_BUFFER_ELEMENT_SIZE;
    ai_float new_data[size];
    ring_buffer_read_data(new_data, new_samples);

    ring_buffer_estimate_gravity(new_samples);
    // initialize velocity to 0
    current_velocity[0] = 0.0f;
    current_velocity[1] = 0.0f;
    current_velocity[2] = 0.0f;

//     const float gravity_x = current_gravity[0];
//     const float gravity_y = current_gravity[1];
//     const float gravity_z = current_gravity[2];
// 
    // TODO: Assuming that the board is flat on the table, the gravity vector is
    // pointing down. We can use this to estimate the velocity of the board.
    //  in the future we can use the gyroscope to correct for the rotation of the board
    const float gravity_x = 0.0f;
    const float gravity_y = 0.0f;
    const float gravity_z = 980.0f;
    const float friction_fudge = 0.98f;

    // integrate the acceleration to get the velocity
    // v = v + a * dt
    // v = v + (a - g) * dt
    // v = v + (a - g) * dt * friction_fudge
    for(int32_t i=0; i < new_samples; i++)
    {
        const ai_float* entry = &new_data[i * RING_BUFFER_ELEMENT_SIZE];
        const float x = (float)entry[0];
        const float y = (float)entry[1];
        const float z = (float)entry[2];
        current_velocity[0] += (x - gravity_x) * dt * friction_fudge;
        current_velocity[1] += (y - gravity_y) * dt * friction_fudge;
        current_velocity[2] += (z - gravity_z) * dt * friction_fudge;
//         _PRINTF("Velocity: %f, %f, %f\r\n", current_velocity[0], current_velocity[1], current_velocity[2]);
    }
//     _PRINTF("Gravity: %f %f %f \r\n", gravity_x, gravity_y, gravity_z);
//     _PRINTF("Velocity: %f, %f, %f\r\n", current_velocity[0], current_velocity[1], current_velocity[2]);
}

int8_t ring_buffer_is_moving(int32_t new_samples, const float threshold)
{
    float velocity_squared = current_velocity[0] * current_velocity[0] +
                             current_velocity[1] * current_velocity[1] +
                             current_velocity[2] * current_velocity[2];

    velocity_squared = sqrtf(velocity_squared);
//     _PRINTF("Velocity squared: %f\r\n", velocity_squared);
//     _PRINTF("\t  Average: %f\r\n", velocity_squared / new_samples);
// 
    return velocity_squared > threshold;
}


float * ring_buffer_get_gravity()
{
    return current_gravity;
}

float * ring_buffer_get_velocity()
{
    return current_velocity;
}

void ring_buffer_reset()
{
    ring_buffer_init();
}


