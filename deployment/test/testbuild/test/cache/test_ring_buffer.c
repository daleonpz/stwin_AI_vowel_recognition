#include "../Core/Src/ring_buffer.c"
#include "/usr/local/bundle/gems/ceedling-0.31.1/vendor/unity/src/unity.h"














void setUp(void)

{

    ring_buffer_init();

}



void tearDown()

{

}



void test_ring_buffer_init()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    uint32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_UINT16)((0)), (UNITY_INT)(UNITY_UINT16)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(28), UNITY_DISPLAY_STYLE_UINT16);



}



void test_ring_buffer_add_one_element()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    uint32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};

    uint32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_UINT16)((0)), (UNITY_INT)(UNITY_UINT16)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(37), UNITY_DISPLAY_STYLE_UINT16);



    ring_buffer_store_data(data);

    index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_UINT16)((1)), (UNITY_INT)(UNITY_UINT16)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(41), UNITY_DISPLAY_STYLE_UINT16);



}



void test_ring_buffer_fill_the_buffer()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    uint32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};

    uint32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_UINT16)((0)), (UNITY_INT)(UNITY_UINT16)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(50), UNITY_DISPLAY_STYLE_UINT16);



    for (int i = 0; i < 20; i++)

    {

        ring_buffer_store_data(data);

    }



    index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_UINT16)((0)), (UNITY_INT)(UNITY_UINT16)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(58), UNITY_DISPLAY_STYLE_UINT16);

}



void test_ring_buffer_fill_the_buffer_plus_one()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    uint32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};

    uint32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_UINT16)((0)), (UNITY_INT)(UNITY_UINT16)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(66), UNITY_DISPLAY_STYLE_UINT16);



    for (int i = 0; i < 20 + 1; i++)

    {

        ring_buffer_store_data(data);

    }



    index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_UINT16)((1)), (UNITY_INT)(UNITY_UINT16)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(74), UNITY_DISPLAY_STYLE_UINT16);

}



void test_ring_buffer_read_one_element()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    int32_t data[6] = {0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};

    int32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((0)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(82), UNITY_DISPLAY_STYLE_INT32);



    ring_buffer_store_data(data);

    index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((1)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(86), UNITY_DISPLAY_STYLE_INT32);



    int32_t data_read[6] = {0};

    ring_buffer_read_data(data_read, 1);

    UnityAssertEqualIntArray(( const void*)((data)), ( const void*)((data_read)), (UNITY_UINT32)(((sizeof(data) / sizeof((data)[0])))), (

   ((void *)0)

   ), (UNITY_UINT)(90), UNITY_DISPLAY_STYLE_INT32, UNITY_ARRAY_TO_ARRAY);

}



void test_ring_buffer_store_10_read_5()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    int32_t data[10][6] = {0};

    for (int i = 0; i < 10; i++)

    {

        for (int j = 0; j < 6; j++)

        {

            data[i][j] = rand();

        }

    }



    int32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((0)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(106), UNITY_DISPLAY_STYLE_INT32);



    for (int i = 0; i < 10; i++)

    {

        ring_buffer_store_data(data[i]);

    }



    index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((10)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(114), UNITY_DISPLAY_STYLE_INT32);



    int32_t data_read[5*6] = {0};

    ring_buffer_read_data(data_read, 5);



    ring_buffer_print_buffer();

    for(int i = 0; i < 5; i++)

    {

       printf("data_read[%d] = %d %d %d %d %d %d \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);

       UnityAssertEqualIntArray(( const void*)((data[9-i])), ( const void*)((data_read + (4-i)*6)), (UNITY_UINT32)(((sizeof(data[i]) / sizeof((data[i])[0])))), (

      ((void *)0)

      ), (UNITY_UINT)(123), UNITY_DISPLAY_STYLE_INT32, UNITY_ARRAY_TO_ARRAY);

    }

}



void test_ring_buffer_store_ring_buffer_size_plus_2_and_read_5()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    int32_t data[20 + 2][6] = {0};

    for (int i = 0; i < 20 + 2; i++)

    {

        for (int j = 0; j < 6; j++)

        {

            data[i][j] = rand();

        }

    }



    int32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((0)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(140), UNITY_DISPLAY_STYLE_INT32);



    for (int i = 0; i < 20 + 2; i++)

    {

        ring_buffer_store_data(data[i]);

    }



    index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((2)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(148), UNITY_DISPLAY_STYLE_INT32);



    int32_t data_read[5*6] = {0};

    ring_buffer_read_data(data_read, 5);



    UnityAssertEqualIntArray(( const void*)((data[20 + 1])), ( const void*)((data_read + 24)), (UNITY_UINT32)(((sizeof(data[0]) / sizeof((data[0])[0])))), (

   ((void *)0)

   ), (UNITY_UINT)(153), UNITY_DISPLAY_STYLE_INT32, UNITY_ARRAY_TO_ARRAY);

    UnityAssertEqualIntArray(( const void*)((data[20])), ( const void*)((data_read + 18)), (UNITY_UINT32)(((sizeof(data[0]) / sizeof((data[0])[0])))), (

   ((void *)0)

   ), (UNITY_UINT)(154), UNITY_DISPLAY_STYLE_INT32, UNITY_ARRAY_TO_ARRAY);

    UnityAssertEqualIntArray(( const void*)((data[20 - 1])), ( const void*)((data_read + 12)), (UNITY_UINT32)(((sizeof(data[0]) / sizeof((data[0])[0])))), (

   ((void *)0)

   ), (UNITY_UINT)(155), UNITY_DISPLAY_STYLE_INT32, UNITY_ARRAY_TO_ARRAY);

    UnityAssertEqualIntArray(( const void*)((data[20 - 2])), ( const void*)((data_read + 6)), (UNITY_UINT32)(((sizeof(data[0]) / sizeof((data[0])[0])))), (

   ((void *)0)

   ), (UNITY_UINT)(156), UNITY_DISPLAY_STYLE_INT32, UNITY_ARRAY_TO_ARRAY);

    UnityAssertEqualIntArray(( const void*)((data[20 - 3])), ( const void*)((data_read)), (UNITY_UINT32)(((sizeof(data[0]) / sizeof((data[0])[0])))), (

   ((void *)0)

   ), (UNITY_UINT)(157), UNITY_DISPLAY_STYLE_INT32, UNITY_ARRAY_TO_ARRAY);



    ring_buffer_print_buffer();

    for(int i = 0; i < 5; i++)

    {

       printf("data_read[%d] = %d %d %d %d %d %d \r\n", i, data_read[i*6], data_read[i*6+1], data_read[i*6+2], data_read[i*6+3], data_read[i*6+4], data_read[i*6+5]);

    }

}



void test_ring_buffer_get_min_max()

{

    printf("---------------------------------------------------------\n");printf("Running: %s\n",__func__);;

    int32_t data[3][6] = { { 1, 2, 3, 4, 5, 6},

                           { 0, 7, 8, -5, 2, 4},

                           { 4, 5, 0, 7, 80, 9}

    };



    int32_t index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((0)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(175), UNITY_DISPLAY_STYLE_INT32);



    for (int i = 0; i < 3; i++)

    {

        ring_buffer_store_data(data[i]);

    }



    index = ring_buffer_get_index();

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((3)), (UNITY_INT)(UNITY_INT32)((index)), (

   ((void *)0)

   ), (UNITY_UINT)(183), UNITY_DISPLAY_STYLE_INT32);



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



    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((0)), (UNITY_INT)(UNITY_INT32)((min_acc)), (

   ((void *)0)

   ), (UNITY_UINT)(199), UNITY_DISPLAY_STYLE_INT32);

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((8)), (UNITY_INT)(UNITY_INT32)((max_acc)), (

   ((void *)0)

   ), (UNITY_UINT)(200), UNITY_DISPLAY_STYLE_INT32);

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((-5)), (UNITY_INT)(UNITY_INT32)((min_gyro)), (

   ((void *)0)

   ), (UNITY_UINT)(201), UNITY_DISPLAY_STYLE_INT32);

    UnityAssertEqualNumber((UNITY_INT)(UNITY_INT32)((80)), (UNITY_INT)(UNITY_INT32)((max_gyro)), (

   ((void *)0)

   ), (UNITY_UINT)(202), UNITY_DISPLAY_STYLE_INT32);



}
