/**
  ******************************************************************************
  * @file    stm32l4xx_it.c 
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   Main Interrupt Service Routines.
  *          This file provides template for all exceptions handler and 
  *          peripherals interrupt service routine.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0055, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0055
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "TargetFeatures.h"
#include "main.h"
#include "stm32l4xx_it.h"

#ifdef PREDMNT1_ENABLE_PRINTF
  extern PCD_HandleTypeDef hpcd;
  extern TIM_HandleTypeDef  TimHandle;
#endif /* PREDMNT1_ENABLE_PRINTF */

//extern EXTI_HandleTypeDef hexti1;
/******************************************************************************/
/*            Cortex-M4 Processor Exceptions Handlers                         */
/******************************************************************************/

/**
  * @brief   This function handles NMI exception.
  * @param  None
  * @retval None
  */
void NMI_Handler(void)
{
}

/**
  * @brief  This function handles Hard Fault exception.
  * @param  None
  * @retval None
  */
void HardFault_Handler(void)
{
  /* Go to infinite loop when Hard Fault exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles Memory Manage exception.
  * @param  None
  * @retval None
  */
void MemManage_Handler(void)
{
  /* Go to infinite loop when Memory Manage exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles Bus Fault exception.
  * @param  None
  * @retval None
  */
void BusFault_Handler(void)
{
  /* Go to infinite loop when Bus Fault exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles Usage Fault exception.
  * @param  None
  * @retval None
  */
void UsageFault_Handler(void)
{
  /* Go to infinite loop when Usage Fault exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles SVCall exception.
  * @param  None
  * @retval None
  */
void SVC_Handler(void)
{
}

/**
  * @brief  This function handles Debug Monitor exception.
  * @param  None
  * @retval None
  */
void DebugMon_Handler(void)
{
}

/**
  * @brief  This function handles PendSVC exception.
  * @param  None
  * @retval None
  */
void PendSV_Handler(void)
{
}

/**
  * @brief  This function handles SysTick Handler.
  * @param  None
  * @retval None
  */
void SysTick_Handler(void)
{
  HAL_IncTick();
}

/******************************************************************************/
/*                 STM32L4xx Peripherals Interrupt Handlers                   */
/*  Add here the Interrupt Handler for the used peripheral(s) (PPP), for the  */
/*  available peripheral interrupt handler's name please refer to the startup */
/*  file (startup_stm32l4xxxx.s).                                             */
/******************************************************************************/

void DMA1_Channel1_IRQHandler(void)
{
  HAL_DMA_IRQHandler(ADC1_Handle.DMA_Handle);
}

void DMA1_Channel4_IRQHandler(void)
{
  HAL_DMA_IRQHandler(AMic_OnBoard_DfsdmFilter.hdmaReg);
}

void DMA1_Channel7_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&DMic_OnBoard_Dma);
  
}

void DFSDM1_FLT0_IRQHandler(void)
{
  HAL_DFSDM_IRQHandler(&DMic_OnBoard_DfsdmFilter);
}

void DFSDM1_FLT1_IRQHandler(void)
{
  HAL_DFSDM_IRQHandler(&AMic_OnBoard_DfsdmFilter);
}

/**
  * @brief  This function handles TIM1 Interrupt request
  * @param  None
  * @retval None
  */
void TIM1_CC_IRQHandler(void)
{
  HAL_TIM_IRQHandler(&TimCCHandle);
}

/**
  * @brief  EXTI4_IRQHandler This function handles External line
  *         interrupt request for ism330dhc.
  * @param  None
  * @retval None
  */
void EXTI4_IRQHandler(void)
{
  HAL_GPIO_EXTI_IRQHandler(M_INT2_O_PIN);
}

/**
* @brief This function handles EXTI line0 interrupt.
*/
void EXTI1_IRQHandler(void)
{
  //HAL_GPIO_EXTI_IRQHandler(HCI_TL_SPI_EXTI_PIN);
  HAL_EXTI_IRQHandler(&H_EXTI_1);
}

/**
  * @brief  EXTI0_IRQHandler This function handles External line
  *         interrupt request for ism330dlc.
  * @param  None
  * @retval None
  */
void EXTI0_IRQHandler(void)
{
    HAL_GPIO_EXTI_IRQHandler(USER_BUTTON_PIN);
}

#ifdef PREDMNT1_ENABLE_PRINTF
/**
  * @brief  This function handles USB-On-The-Go FS global interrupt request.
  * @param  None
  * @retval None
  */
void OTG_FS_IRQHandler(void)
{
  HAL_PCD_IRQHandler(&hpcd);
}

/**
  * @brief  This function handles TIM interrupt request.
  * @param  None
  * @retval None
  */
void TIM2_IRQHandler(void)
{
  HAL_TIM_IRQHandler(&TimHandle);
}
#endif /* PREDMNT1_ENABLE_PRINTF */

/************************ (C) COPYRIGHT 2016 STMicroelectronics *****END OF FILE****/
