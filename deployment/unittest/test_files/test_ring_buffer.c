
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
    uint32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
    uint32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (0, index);

    ring_buffer_store_data(data);
    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_UINT16 (1, index);

}

void test_ring_buffer_fill_the_buffer()
{
    PRINT_TEST_HEADER;
    uint32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
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
    uint32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
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
    int32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
    int32_t index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (0, index);

    ring_buffer_store_data(data);
    index = ring_buffer_get_index();
    TEST_ASSERT_EQUAL_INT32 (1, index);

    int32_t data_read[6] = {0};
    ring_buffer_read_data(data_read, 1);
    TEST_ASSERT_EQUAL_INT32_ARRAY (data, data_read, NELEMS(data));
}

void test_ring_buffer_store_10_read_5()
{
    PRINT_TEST_HEADER;
    int32_t data[10][6] = {0};
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

    int32_t data_read[5*6] = {0};
    ring_buffer_read_data(data_read, 5);

    ring_buffer_print_buffer();
    for(int i = 0; i < 5; i++)
    {
       printf("data_read[%d] = %d %d %d %d %d %d \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);
       TEST_ASSERT_EQUAL_INT32_ARRAY (data[9-i], data_read + (4-i)*6, NELEMS(data[i]));
    }
}

void test_ring_buffer_store_ring_buffer_size_plus_2_and_read_5()
{
    PRINT_TEST_HEADER;
    int32_t data[RING_BUFFER_SIZE + 2][6] = {0};
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

    int32_t data_read[5*6] = {0};
    ring_buffer_read_data(data_read, 5);

    TEST_ASSERT_EQUAL_INT32_ARRAY (data[RING_BUFFER_SIZE + 1], data_read + 24 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_INT32_ARRAY (data[RING_BUFFER_SIZE], data_read + 18 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_INT32_ARRAY (data[RING_BUFFER_SIZE - 1], data_read + 12 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_INT32_ARRAY (data[RING_BUFFER_SIZE - 2], data_read + 6 , NELEMS(data[0]));
    TEST_ASSERT_EQUAL_INT32_ARRAY (data[RING_BUFFER_SIZE - 3], data_read , NELEMS(data[0]));

    ring_buffer_print_buffer();
    for(int i = 0; i < 5; i++)
    {
       printf("data_read[%d] = %d %d %d %d %d %d \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);
    }
}

void test_ring_buffer_get_min_max()
{
    PRINT_TEST_HEADER;
    int32_t data[3][6] = { { 1, 2, 3, 4,  5,  6},
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

    int32_t data_read[3*6] = {0};
    ring_buffer_read_data(data_read, 3);

    int32_t min_acc = 0;
    int32_t min_gyro = 0;
    int32_t max_acc = 0;
    int32_t max_gyro = 0;

    for( int i=0; i<3; i++){
        printf("data_read[%d] = %d %d %d %d %d %d \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);
    }
    ring_buffer_get_min(&min_acc, &min_gyro);
    ring_buffer_get_max(&max_acc, &max_gyro);

    TEST_ASSERT_EQUAL_INT32 (0, min_acc);
    TEST_ASSERT_EQUAL_INT32 (8, max_acc);
    TEST_ASSERT_EQUAL_INT32 (-5, min_gyro);
    TEST_ASSERT_EQUAL_INT32 (80, max_gyro);

}