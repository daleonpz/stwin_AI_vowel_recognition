/**
  ******************************************************************************
  * @file    PREDMNT1_config.h
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   FP-IND-PREDMNT1 configuration
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
#ifndef __PREDMNT1_CONFIG_H
#define __PREDMNT1_CONFIG_H

/*************** Debug Defines ******************/
/* For enabling the printf on UART */
#define PREDMNT1_ENABLE_PRINTF        (1)
// #define PREDMNT1_ACTIVATE_PRINTF        (1)

/* For enabling connection and notification subscriptions debug */
// #define PREDMNT1_DEBUG_CONNECTION      (1)

/* For enabling trasmission for notified services (except for quaternions) */
// #define PREDMNT1_DEBUG_NOTIFY_TRAMISSION   (1)

/*************** Don't Change the following defines *************/

/* Package Version only numbers 0->9 */
#define PREDMNT1_VERSION_MAJOR '2'
#define PREDMNT1_VERSION_MINOR '4'
#define PREDMNT1_VERSION_PATCH '0'

/* Package Name */
#define PREDMNT1_PACKAGENAME "FP-IND-PREDMNT1"
#define CONFIG_NAME "Application - Predictive Maintenance"

/* Environmental Sensor Istance */
#define TEMPERATURE_INSTANCE_1  HTS221_0
#define HUMIDITY_INSTANCE       HTS221_0
#define TEMPERATURE_INSTANCE_2  LPS22HH_0
#define PRESSURE_INSTANCE       LPS22HH_0

/* Motion Sensor Istance */
// #define ACCELERO_INSTANCE       IIS2DH_0
#define ACCELERO_INSTANCE       ISM330DHCX_0
#define GYRO_INSTANCE           ISM330DHCX_0
#define MAGNETO_INSTANCE        IIS2MDC_0

/* Environmental Sensor API */
#define ENV_SENSOR_Init         BSP_ENV_SENSOR_Init
#define ENV_SENSOR_Enable       BSP_ENV_SENSOR_Enable
#define ENV_SENSOR_GetValue     BSP_ENV_SENSOR_GetValue

/* Motion Sensor API */
#define MOTION_SENSOR_Init                      BSP_MOTION_SENSOR_Init
#define MOTION_SENSOR_Enable                    BSP_MOTION_SENSOR_Enable

#define MOTION_SENSOR_AxesRaw_t                 BSP_MOTION_SENSOR_AxesRaw_t
#define MOTION_SENSOR_Axes_t                    BSP_MOTION_SENSOR_Axes_t

#define MOTION_SENSOR_GetAxes                   BSP_MOTION_SENSOR_GetAxes

#define MOTION_SENSOR_GetSensitivity            BSP_MOTION_SENSOR_GetSensitivity
#define MOTION_SENSOR_SetFullScale              BSP_MOTION_SENSOR_SetFullScale

#define MOTION_SENSOR_Write_Register            BSP_MOTION_SENSOR_Write_Register          

#define MOTION_SENSOR_SetOutputDataRate         BSP_MOTION_SENSOR_SetOutputDataRate
#define MOTION_SENSOR_Enable_HP_Filter          BSP_MOTION_SENSOR_Enable_HP_Filter
#define MOTION_SENSOR_Set_INT2_DRDY             BSP_MOTION_SENSOR_Set_INT2_DRDY
#define MOTION_SENSOR_DRDY_Set_Mode             BSP_MOTION_SENSOR_DRDY_Set_Mode

#define MOTION_SENSOR_FIFO_Set_Mode             BSP_MOTION_SENSOR_FIFO_Set_Mode
#define MOTION_SENSOR_FIFO_Set_INT2_FIFO_Full   BSP_MOTION_SENSOR_FIFO_Set_INT2_FIFO_Full
#define MOTION_SENSOR_FIFO_Read                 BSP_MOTION_SENSOR_FIFO_Read
#define MOTION_SENSOR_FIFO_Get_Data_Word        BSP_MOTION_SENSOR_FIFO_Get_Data_Word
#define MOTION_SENSOR_FIFO_Set_BDR              BSP_MOTION_SENSOR_FIFO_Set_BDR
#define MOTION_SENSOR_FIFO_Set_Watermark_Level  BSP_MOTION_SENSOR_FIFO_Set_Watermark_Level
#define MOTION_SENSOR_FIFO_Set_Stop_On_Fth      BSP_MOTION_SENSOR_FIFO_Set_Stop_On_Fth 

/*****************
* Sensor Setting *
******************/

/* ISM330DLCX HPF Configuration */
#define HPF_ODR_DIV_400         ISM330DHCX_HP_ODR_DIV_400

/*************************
* Serial control section *
**************************/
#ifdef PREDMNT1_ENABLE_PRINTF
#include "usbd_cdc_interface.h"
#define _PRINTF(...) {\
  char TmpBufferToWrite[256];\
  int32_t TmpBytesToWrite;\
  TmpBytesToWrite = sprintf( TmpBufferToWrite, __VA_ARGS__);\
  CDC_Fill_Buffer(( uint8_t * )TmpBufferToWrite, (uint32_t)TmpBytesToWrite);\
}
#endif


#ifdef PREDMNT1_ACTIVATE_PRINTF
    #define PREDMNT1_PRINTF(...) _PRINTF(__VA_ARGS__)
#else /* PREDMNT1_ENABLE_PRINTF */
  #define PREDMNT1_PRINTF(...)
#endif /* PREDMNT1_ENABLE_PRINTF */



/* STM32 Unique ID */
#define STM32_UUID ((uint32_t *)0x1FFF7590)

/* STM32 MCU_ID */
#define STM32_MCU_ID ((uint32_t *)0xE0042000)
/* Control Section */

#endif /* __PREDMNT1_CONFIG_H */

/******************* (C) COPYRIGHT 2021 STMicroelectronics *****END OF FILE****/
