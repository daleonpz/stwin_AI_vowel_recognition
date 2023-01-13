#!/bin/sh

# This script creates the C files for the STM32F4xx HAL library
# It is based on the STM32CubeMX tool

STM32AI='/home/me/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.1.0/Utilities/linux/stm32ai'
ONNX_MODEL='results/model.onnx'
NAME_PROJECT='model'
OUTPUT_DIR='../deployment/X-CUBE-AI/App'

${STM32AI} generate -m ${ONNX_MODEL} --type onnx -o ${OUTPUT_DIR} --name ${NAME_PROJECT}


