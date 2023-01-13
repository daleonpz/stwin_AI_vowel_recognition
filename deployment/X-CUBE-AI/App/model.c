/**
  ******************************************************************************
  * @file    model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Jan 13 14:25:59 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "model.h"
#include "model_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"




#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_model
 
#undef AI_MODEL_MODEL_SIGNATURE
#define AI_MODEL_MODEL_SIGNATURE     "90a229c1150941d5287056359126c448"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Fri Jan 13 14:25:59 2023"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_MODEL_N_BATCHES
#define AI_MODEL_N_BATCHES         (1)




/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2400, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2400, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1200, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _fc_layers_layers_0_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _fc_layers_layers_1_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _fc_layers_layers_2_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  output_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 648, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2592, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  fc_layers_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  fc_layers_2_weight_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  fc_layers_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  fc_layers_0_weight_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4800, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 480, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 480, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 20, 6), AI_STRIDE_INIT(4, 4, 4, 80, 1600),
  1, &input_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_Transpose_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 20, 20), AI_STRIDE_INIT(4, 4, 4, 24, 480),
  1, &input_Transpose_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 10, 10), AI_STRIDE_INIT(4, 4, 4, 48, 480),
  1, &_conv1_conv1_0_Conv_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 5, 5), AI_STRIDE_INIT(4, 4, 4, 96, 480),
  1, &_conv2_conv2_0_Conv_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_output0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 600, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2400, 2400),
  1, &_conv2_conv2_0_Conv_output_0_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _fc_layers_layers_0_Gemm_output_0_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_fc_layers_layers_0_Gemm_output_0_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _fc_layers_layers_1_Relu_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_fc_layers_layers_1_Relu_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _fc_layers_layers_2_Gemm_output_0_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &_fc_layers_layers_2_Gemm_output_0_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  output_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &output_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_weights, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 6, 3, 3, 12), AI_STRIDE_INIT(4, 4, 24, 72, 216),
  1, &_conv1_conv1_0_Conv_output_0_weights_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &_conv1_conv1_0_Conv_output_0_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 12, 3, 3, 24), AI_STRIDE_INIT(4, 4, 48, 144, 432),
  1, &_conv2_conv2_0_Conv_output_0_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &_conv2_conv2_0_Conv_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  fc_layers_2_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &fc_layers_2_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  fc_layers_2_weight, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 2), AI_STRIDE_INIT(4, 4, 32, 64, 64),
  1, &fc_layers_2_weight_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  fc_layers_0_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &fc_layers_0_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  fc_layers_0_weight, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 600, 1, 8), AI_STRIDE_INIT(4, 4, 2400, 19200, 19200),
  1, &fc_layers_0_weight_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_scratch0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 20, 2), AI_STRIDE_INIT(4, 4, 4, 48, 960),
  1, &_conv1_conv1_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 2), AI_STRIDE_INIT(4, 4, 4, 96, 960),
  1, &_conv2_conv2_0_Conv_output_0_scratch0_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_layers_layers_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &output_chain,
  NULL, &output_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_layers_layers_2_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_fc_layers_layers_1_Relu_output_0_output, &fc_layers_2_weight, &fc_layers_2_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_layers_layers_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_layers_layers_2_Gemm_output_0_layer, 9,
  GEMM_TYPE, 0x0, NULL,
  gemm, forward_gemm,
  &_fc_layers_layers_2_Gemm_output_0_chain,
  NULL, &output_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 1, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_layers_layers_1_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_layers_layers_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_layers_layers_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_layers_layers_1_Relu_output_0_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_fc_layers_layers_1_Relu_output_0_chain,
  NULL, &_fc_layers_layers_2_Gemm_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_layers_layers_0_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv2_conv2_0_Conv_output_0_output0, &fc_layers_0_weight, &fc_layers_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_layers_layers_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_layers_layers_0_Gemm_output_0_layer, 7,
  GEMM_TYPE, 0x0, NULL,
  gemm, forward_gemm,
  &_fc_layers_layers_0_Gemm_output_0_chain,
  NULL, &_fc_layers_layers_1_Relu_output_0_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 1, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_conv1_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_conv2_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv2_conv2_0_Conv_output_0_weights, &_conv2_conv2_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv2_conv2_0_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _conv2_conv2_0_Conv_output_0_layer, 5,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &_conv2_conv2_0_Conv_output_0_chain,
  NULL, &_fc_layers_layers_0_Gemm_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_conv1_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv1_conv1_0_Conv_output_0_weights, &_conv1_conv1_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_conv1_0_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _conv1_conv1_0_Conv_output_0_layer, 2,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &_conv1_conv1_0_Conv_output_0_chain,
  NULL, &_conv2_conv2_0_Conv_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_Transpose_chain,
  NULL, &_conv1_conv1_0_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 32408, 1, 1),
    32408, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12624, 1, 1),
    12624, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_OUT_NUM, &output_output),
  &input_Transpose_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 32408, 1, 1),
      32408, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12624, 1, 1),
      12624, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODEL_OUT_NUM, &output_output),
  &input_Transpose_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  ai_ptr activations_map[1] = AI_C_ARRAY_INIT;

  if (ai_platform_get_activations_map(activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    input_Transpose_output_array.data = AI_PTR(activations_map[0] + 1104);
    input_Transpose_output_array.data_start = AI_PTR(activations_map[0] + 1104);
    _conv1_conv1_0_Conv_output_0_scratch0_array.data = AI_PTR(activations_map[0] + 10704);
    _conv1_conv1_0_Conv_output_0_scratch0_array.data_start = AI_PTR(activations_map[0] + 10704);
    _conv1_conv1_0_Conv_output_0_output_array.data = AI_PTR(activations_map[0] + 0);
    _conv1_conv1_0_Conv_output_0_output_array.data_start = AI_PTR(activations_map[0] + 0);
    _conv2_conv2_0_Conv_output_0_scratch0_array.data = AI_PTR(activations_map[0] + 4800);
    _conv2_conv2_0_Conv_output_0_scratch0_array.data_start = AI_PTR(activations_map[0] + 4800);
    _conv2_conv2_0_Conv_output_0_output_array.data = AI_PTR(activations_map[0] + 6720);
    _conv2_conv2_0_Conv_output_0_output_array.data_start = AI_PTR(activations_map[0] + 6720);
    _fc_layers_layers_0_Gemm_output_0_output_array.data = AI_PTR(activations_map[0] + 0);
    _fc_layers_layers_0_Gemm_output_0_output_array.data_start = AI_PTR(activations_map[0] + 0);
    _fc_layers_layers_1_Relu_output_0_output_array.data = AI_PTR(activations_map[0] + 32);
    _fc_layers_layers_1_Relu_output_0_output_array.data_start = AI_PTR(activations_map[0] + 32);
    _fc_layers_layers_2_Gemm_output_0_output_array.data = AI_PTR(activations_map[0] + 0);
    _fc_layers_layers_2_Gemm_output_0_output_array.data_start = AI_PTR(activations_map[0] + 0);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool model_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  ai_ptr weights_map[1] = AI_C_ARRAY_INIT;

  if (ai_platform_get_weights_map(weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _conv1_conv1_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv1_conv1_0_Conv_output_0_weights_array.data = AI_PTR(weights_map[0] + 0);
    _conv1_conv1_0_Conv_output_0_weights_array.data_start = AI_PTR(weights_map[0] + 0);
    _conv1_conv1_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv1_conv1_0_Conv_output_0_bias_array.data = AI_PTR(weights_map[0] + 2592);
    _conv1_conv1_0_Conv_output_0_bias_array.data_start = AI_PTR(weights_map[0] + 2592);
    _conv2_conv2_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv2_conv2_0_Conv_output_0_weights_array.data = AI_PTR(weights_map[0] + 2640);
    _conv2_conv2_0_Conv_output_0_weights_array.data_start = AI_PTR(weights_map[0] + 2640);
    _conv2_conv2_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv2_conv2_0_Conv_output_0_bias_array.data = AI_PTR(weights_map[0] + 13008);
    _conv2_conv2_0_Conv_output_0_bias_array.data_start = AI_PTR(weights_map[0] + 13008);
    fc_layers_2_bias_array.format |= AI_FMT_FLAG_CONST;
    fc_layers_2_bias_array.data = AI_PTR(weights_map[0] + 13104);
    fc_layers_2_bias_array.data_start = AI_PTR(weights_map[0] + 13104);
    fc_layers_2_weight_array.format |= AI_FMT_FLAG_CONST;
    fc_layers_2_weight_array.data = AI_PTR(weights_map[0] + 13112);
    fc_layers_2_weight_array.data_start = AI_PTR(weights_map[0] + 13112);
    fc_layers_0_bias_array.format |= AI_FMT_FLAG_CONST;
    fc_layers_0_bias_array.data = AI_PTR(weights_map[0] + 13176);
    fc_layers_0_bias_array.data_start = AI_PTR(weights_map[0] + 13176);
    fc_layers_0_weight_array.format |= AI_FMT_FLAG_CONST;
    fc_layers_0_weight_array.data = AI_PTR(weights_map[0] + 13208);
    fc_layers_0_weight_array.data_start = AI_PTR(weights_map[0] + 13208);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_model_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MODEL_MODEL_NAME,
      .model_signature   = AI_MODEL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 537700,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_model_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MODEL_MODEL_NAME,
      .model_signature   = AI_MODEL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 537700,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_model_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_model_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_model_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_model_create(network, AI_MODEL_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_model_data_params_get(&params) != true) {
        err = ai_model_get_error(*network);
        return err;
    }
#if defined(AI_MODEL_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_MODEL_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_model_init(*network, &params) != true) {
        err = ai_model_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_model_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_model_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_model_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_model_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= model_configure_weights(net_ctx, params);
  ok &= model_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_model_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_model_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_MODEL_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

