
#include <unity.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h> 

#include "ring_buffer.c"

#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))
#define PRINT_TEST_HEADER  printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__); 

void setUp(void)
{
    ring_buffer_init();
}

void tearDown()
{
}

void test_ring_buffer_init()
{
    PRINT_TEST_HEADER;
    uint32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (0, index); 

}

void test_ring_buffer_add_one_element()
{
    PRINT_TEST_HEADER;
    float data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
    uint32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (0, index);

    ring_buffer_store_data(data);
    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (1, index);

}

void test_ring_buffer_fill_the_buffer()
{
    PRINT_TEST_HEADER;
    float data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
    uint32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (0, index);

    for (int i = 0; i < RING_BUFFER_SIZE; i++)
    {
        ring_buffer_store_data(data);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (0, index);
}

void test_ring_buffer_fill_the_buffer_plus_one()
{
    PRINT_TEST_HEADER;
    float data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
    uint32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (0, index);

    for (int i = 0; i < RING_BUFFER_SIZE + 1; i++)
    {
        ring_buffer_store_data(data);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (1, index);
}

void test_ring_buffer_read_one_element()
{
    PRINT_TEST_HEADER;
    float data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    ring_buffer_store_data(data);
    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (1, index);

    float data_read[6] = {0};
    ring_buffer_read_data(data_read, 1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY (data, data_read, NELEMS(data));
}

void test_ring_buffer_store_10_read_5()
{
    PRINT_TEST_HEADER;
    float data[10][6] = {0};
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            data[i][j] = rand();
        }
    }

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < 10; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (10, index);

    float data_read[5*6] = {0};
    ring_buffer_read_data(data_read, 5);

//     ring_buffer_print_buffer();
    for(int i = 0; i < 5; i++)
    {
       printf("data_read[%d] = %f %f %f %f %f %f \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);
       TEST_ASSERT_EQUAL_FLOAT_ARRAY (data[9-i], data_read + (4-i)*6, NELEMS(data[i]));
    }
}

void test_ring_buffer_store_ring_buffer_size_plus_2_and_read_5()
{
    PRINT_TEST_HEADER;
    float data[RING_BUFFER_SIZE + 2][6] = {0};
    for (int i = 0; i < RING_BUFFER_SIZE + 2; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            data[i][j] = rand();
        }
    }

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < RING_BUFFER_SIZE + 2; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (2, index);

    float data_read[5*6] = {0};
    ring_buffer_read_data(data_read, 5);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY (data[RING_BUFFER_SIZE + 1], data_read + 24 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY (data[RING_BUFFER_SIZE], data_read + 18 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY (data[RING_BUFFER_SIZE - 1], data_read + 12 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY (data[RING_BUFFER_SIZE - 2], data_read + 6 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_FLOAT_ARRAY (data[RING_BUFFER_SIZE - 3], data_read , NELEMS(data[0]));

//     ring_buffer_print_buffer();
    for(int i = 0; i < 5; i++)
    {
       printf("data_read[%d] = %f %f %f %f %f %f \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);
    }
}

void test_ring_buffer_get_min_max()
{
    PRINT_TEST_HEADER;
    float data[3][6] = { { 1, 2, 3, 4,  5,  6},
                           { 0, 7, 8, -5, 2,  4},
                           { 4, 5, 0, 7,  80, 9}
    };

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < 3; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (3, index);

    float data_read[3*6] = {0};
    ring_buffer_read_data(data_read, 3);

    float min_acc = 0;
    float min_gyro = 0;
    float max_acc = 0;
    float max_gyro = 0;

    for( int i=0; i<3; i++){
        printf("data_read[%d] = %f %f %f %f %f %f \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);
    }
    ring_buffer_get_min(&min_acc, &min_gyro);
    ring_buffer_get_max(&max_acc, &max_gyro);

    printf("min_acc = %f, min_gyro = %f, max_acc = %f, max_gyro = %f \r\n", min_acc, min_gyro, max_acc, max_gyro);
    TEST_ASSERT_EQUAL_FLOAT (0, min_acc);
    TEST_ASSERT_EQUAL_FLOAT (8, max_acc);
    TEST_ASSERT_EQUAL_FLOAT (-5, min_gyro);
    TEST_ASSERT_EQUAL_FLOAT (80, max_gyro);
}


void test_ring_buffer_estimate_gravity_simple()
{
    PRINT_TEST_HEADER;
    float data[3][6] = { { 1, 2, 3, 4,  5,  6},
                           { 0, 7, -8, -5, 2,  4},
                           { 4, 5, 0, 7,  80, 9}
    };

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < 3; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (3, index);

    ring_buffer_estimate_gravity(3);

    float * gravity_ptr = ring_buffer_get_gravity();

    TEST_ASSERT_EQUAL_FLOAT(5/3.0, gravity_ptr[0]);
    TEST_ASSERT_EQUAL_FLOAT(14/3.0, gravity_ptr[1]);
    TEST_ASSERT_EQUAL_FLOAT(-5/3.0, gravity_ptr[2]);

}

void test_ring_buffer_estimate_gravity_store_plus_2_and_read_5()
{
    PRINT_TEST_HEADER;
    float data[RING_BUFFER_SIZE + 2][6] = {0};
    for (int i = 0; i < RING_BUFFER_SIZE + 2; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            data[i][j] = rand();
        }
    }

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < RING_BUFFER_SIZE + 2; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    ring_buffer_estimate_gravity(5);

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (2, index);

    float * gravity_ptr = ring_buffer_get_gravity();

    TEST_ASSERT_EQUAL_FLOAT((data[RING_BUFFER_SIZE + 1][0] + 
                data[RING_BUFFER_SIZE][0] + 
                data[RING_BUFFER_SIZE - 1][0] + 
                data[RING_BUFFER_SIZE - 2][0] + 
                data[RING_BUFFER_SIZE - 3][0])/5.0, 
            gravity_ptr[0]);
    TEST_ASSERT_EQUAL_FLOAT((data[RING_BUFFER_SIZE + 1][1] + 
                data[RING_BUFFER_SIZE][1] + 
                data[RING_BUFFER_SIZE - 1][1] + 
                data[RING_BUFFER_SIZE - 2][1] + 
                data[RING_BUFFER_SIZE - 3][1])/5.0,
            gravity_ptr[1]);
    TEST_ASSERT_EQUAL_FLOAT((data[RING_BUFFER_SIZE + 1][2] + 
                data[RING_BUFFER_SIZE][2] + 
                data[RING_BUFFER_SIZE - 1][2] + 
                data[RING_BUFFER_SIZE - 2][2] + 
                data[RING_BUFFER_SIZE - 3][2])/5.0, 
            gravity_ptr[2]);
}

void test_ring_buffer_estimate_velocity_simple()
{
    PRINT_TEST_HEADER;

    float data[3][6] = { { 1, 2, 3, 4,  5,  6},
                           { 0, 7, -8, -5, 2,  4},
                           { 4, 5, 0, 7,  80, 9}
    };

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < 3; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (3, index);

    ring_buffer_estimate_velocity(3, 1/200.0);

    float * velocity_ptr = ring_buffer_get_velocity();

    float * gravity_ptr = ring_buffer_get_gravity();

    float expected_velocity[3] = {0};
    expected_velocity[0] = ( data[0][0] - gravity_ptr[0] + data[1][0] - gravity_ptr[0] + data[2][0] - gravity_ptr[0] );
    expected_velocity[0] = expected_velocity[0] * 1/200.0 * 0.98f;
    expected_velocity[1] = ( data[0][1] - gravity_ptr[1] + data[1][1] - gravity_ptr[1] + data[2][1] - gravity_ptr[1] );
    expected_velocity[1] = expected_velocity[1] * 1/200.0 * 0.98f;
    expected_velocity[2] = ( data[0][2] - gravity_ptr[2] + data[1][2] - gravity_ptr[2] + data[2][2] - gravity_ptr[2] );
    expected_velocity[2] = expected_velocity[2] * 1/200.0 * 0.98f;

    TEST_ASSERT_EQUAL_FLOAT( expected_velocity[0], velocity_ptr[0]);
    TEST_ASSERT_EQUAL_FLOAT( expected_velocity[1], velocity_ptr[1]);
    TEST_ASSERT_EQUAL_FLOAT( expected_velocity[2], velocity_ptr[2]);
}

void test_ring_buffer_is_moving_true(void)
{
    PRINT_TEST_HEADER;
    float data[3][6] = { { 1, 2, 3, 4,  5,  6},
                           { 0, 7, -8, -5, 2,  4},
                           { 4, 5, 0, 7,  80, 9}
    };

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < 3; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (3, index);

    float * velocity_ptr = ring_buffer_get_velocity();

    velocity_ptr[0] = 1.1;
    velocity_ptr[1] = 2.1;
    velocity_ptr[2] = 0.1;

    TEST_ASSERT_TRUE(ring_buffer_is_moving(3));
}


void test_ring_buffer_is_moving_false(void)
{
    PRINT_TEST_HEADER;
    float data[3][6] = { { 1, 2, 3, 4,  5,  6},
                           { 0, 7, -8, -5, 2,  4},
                           { 4, 5, 0, 7,  80, 9}
    };

    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    for (int i = 0; i < 3; i++)
    {
        ring_buffer_store_data(data[i]);
    }

    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (3, index);

    float * velocity_ptr = ring_buffer_get_velocity();

    velocity_ptr[0] = 0.001;
    velocity_ptr[1] = 0.001;
    velocity_ptr[2] = 0.001;

    TEST_ASSERT_FALSE(ring_buffer_is_moving(3));

}
