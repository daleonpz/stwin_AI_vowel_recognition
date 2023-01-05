/**
  ******************************************************************************
  * @file    hci_tl_interface.h
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   This file contains all the functions prototypes for the STM32
  *          BlueNRG2 HCI Transport Layer interface
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
#ifndef __HCI_TL_INTERFACE_H
#define __HCI_TL_INTERFACE_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "STWIN_bus.h"

/** @addtogroup Projects
  * @{
  */

/** @addtogroup DEMONSTRATIONS Demonstrations
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE Predictive Maintenance
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE_HCI_TL_INTERFACE Predictive Maintenance hci tl interface
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE_HCI_TL_INTERFACE_EXPORTED_DEFINES Predictive Maintenance hci tl interface Exported Defines
 * @{
 */

/* Exported Defines ----------------------------------------------------------*/

#define HCI_TL_SPI_EXTI_PORT  GPIOG
#define HCI_TL_SPI_EXTI_PIN   GPIO_PIN_1
#define HCI_TL_SPI_EXTI_IRQn  EXTI1_IRQn

#define HCI_TL_SPI_IRQ_PORT   GPIOG
#define HCI_TL_SPI_IRQ_PIN    GPIO_PIN_1

#define HCI_TL_SPI_CS_PORT    GPIOG
#define HCI_TL_SPI_CS_PIN     GPIO_PIN_5

#define HCI_TL_RST_PORT       GPIOA
#define HCI_TL_RST_PIN        GPIO_PIN_8

/**
  * @}
  */

   
/** @addtogroup PREDCTIVE_MAINTENANCE_HCI_TL_INTERFACE_EXPORTED_IO_BUS_FUNCTIONS Predictive Maintenance hci tl interface Exported IO Bus Functions
 * @{
 */

/* Exported variables --------------------------------------------------------*/
extern EXTI_HandleTypeDef     hexti1;
#define H_EXTI_1 hexti1

/**
  * @}
  */


/** @addtogroup PREDCTIVE_MAINTENANCE_HCI_TL_INTERFACE_EXPORTED_IO_BUS_FUNCTIONS_PROTOTYPES Predictive Maintenance hci tl interface Exported IO Bus Functions Prototypes
 * @{
 */

/* Exported Functions --------------------------------------------------------*/
int32_t HCI_TL_SPI_Init    (void* pConf);
int32_t HCI_TL_SPI_DeInit  (void);
int32_t HCI_TL_SPI_Receive (uint8_t* buffer, uint16_t size);
int32_t HCI_TL_SPI_Send    (uint8_t* buffer, uint16_t size);
int32_t HCI_TL_SPI_Reset   (void);

/**
  * @}
  */
 
 /** @addtogroup PREDCTIVE_MAINTENANCE_HCI_TL_INTERFACE_EXPORTED_MAIN_FUNCTIONS_PROTOTYPES Predictive Maintenance hci tl interface Exported Main Functions Prototypes
 * @{
 */
 
/**
 * @brief  Register hci_tl_interface IO bus services
 *
 * @param  None
 * @retval None
 */
void hci_tl_lowlevel_init(void);

/**
 * @brief HCI Transport Layer Low Level Interrupt Service Routine
 *
 * @param  None
 * @retval None
 */
void hci_tl_lowlevel_isr(void);

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif
#endif /* __HCI_TL_INTERFACE_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
