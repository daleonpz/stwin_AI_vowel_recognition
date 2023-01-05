/**
  ******************************************************************************
  * @file    usbd_cdc_interface.h
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   Header for usbd_cdc_interface.c file.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __USBD_CDC_IF_H
#define __USBD_CDC_IF_H

/* Includes ------------------------------------------------------------------*/
#include "usbd_cdc.h"

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
#define USB_RxBufferDim                         2048

/* Definition for TIMx clock resources */
//#define TIMx                             TIM3
//#define TIMx_CLK_ENABLE                  __HAL_RCC_TIM3_CLK_ENABLE
//#define TIMx_FORCE_RESET()               __HAL_RCC_USART3_FORCE_RESET()
//#define TIMx_RELEASE_RESET()             __HAL_RCC_USART3_RELEASE_RESET()
//
///* Definition for TIMx's NVIC */
//#define TIMx_IRQn                        TIM3_IRQn
//#define TIMx_IRQHandler                  TIM3_IRQHandler

#define TIMx                             TIM2
#define TIMx_CLK_ENABLE                  __HAL_RCC_TIM2_CLK_ENABLE
#define TIMx_FORCE_RESET()               __HAL_RCC_USART2_FORCE_RESET()
#define TIMx_RELEASE_RESET()             __HAL_RCC_USART2_RELEASE_RESET()

/* Definition for TIMx's NVIC */
#define TIMx_IRQn                        TIM2_IRQn
#define TIMx_IRQHandler                  TIM2_IRQHandler

/* Periodically, the state of the buffer "UserTxBuffer" is checked.
   The period depends on CDC_POLLING_INTERVAL */
#define CDC_POLLING_INTERVAL             5 /* in ms. The max is 65 and the min is 1 */

/* Exported Variables -------------------------------------------------------------*/
extern USBD_CDC_ItfTypeDef  USBD_CDC_fops;

/* TIM handler declaration */
extern TIM_HandleTypeDef  TimHandle;

/* Exported macro ------------------------------------------------------------*/

/* Exported functions ------------------------------------------------------- */
uint8_t CDC_Fill_Buffer(uint8_t* Buf, uint32_t TotalLen);
void CDC_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim);

#endif /* __USBD_CDC_IF_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
