 /**
  ******************************************************************************
  * @file    main.h 
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   Header for main.c module
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
#ifndef __MAIN_H
#define __MAIN_H

/* Includes ------------------------------------------------------------------*/
#include "PREDMNT1_config.h"
#include "hci_tl_interface.h"
#include "MotionSP_Manager.h"
// #include "BLE_Manager.h"
#include "stm32l4xx_hal.h"

/** @addtogroup Projects
  * @{
  */

/** @addtogroup DEMONSTRATIONS Demonstrations
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE Predictive Maintenance BLE
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE_MAIN Predictive Maintenance main
  * @{
  */

/** @defgroup PREDCTIVE_MAINTENANCE_MAIN_EXPORTED_TYPES Predictive Maintenance Main Exported Types
  * @{
  */

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_MAIN_EXPORTED_MACRO Predictive Maintenance Main Exported Macro
  * @{
  */

/* Exported macro ------------------------------------------------------------*/
#define MCR_BLUEMS_F2I_1D(in, out_int, out_dec) {out_int = (int32_t)in; out_dec= (int32_t)((in-out_int)*10);};
#define MCR_BLUEMS_F2I_2D(in, out_int, out_dec) {out_int = (int32_t)in; out_dec= (int32_t)((in-out_int)*100);};

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_MAIN_EXPORTED_FUNCTIONS_PROTOTYPES Predictive Maintenance Main Exported Functions Prototypes
  * @{
  */

/* Exported functions ------------------------------------------------------- */
extern void Error_Handler(void);
extern void ReadEnvironmentalData(int32_t *PressToSend,uint16_t *HumToSend,int16_t *Temp1ToSend,int16_t *Temp2ToSend);

extern unsigned char SaveVibrationParamToMemory(void);

uint8_t getBlueNRG2_Version(uint8_t *hwVersion, uint16_t *fwVersion);

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_MAIN_EXPORTED_VARIABLES Predictive Maintenance Main Exported Variables
  * @{
  */

/* Exported Variables ------------------------------------------------------- */
extern volatile uint32_t HCI_ProcessEvent;
extern volatile uint8_t FifoEnabled;

extern TIM_HandleTypeDef TimCCHandle;

extern uint8_t EnvironmentalTimerEnabled;
extern uint8_t AudioLevelTimerEnabled;
extern uint8_t BatteryTimerEnabled;

extern uint8_t AudioLevelEnable;

extern uint32_t uhCCR1_Val;
extern uint32_t uhCCR2_Val;
extern uint32_t uhCCR3_Val;
extern uint32_t uhCCR4_Val;

extern uint8_t NodeName[];

extern float RMS_Ch[];
extern float DBNOISE_Value_Old_Ch[];

extern volatile uint32_t PredictiveMaintenance;

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_MAIN_EXPORTED_DEFINES Predictive Maintenance Main Exported Defines
  * @{
  */

/* Exported defines --------------------------------------------------------- */

    /* Update frequency for environmental sensor [Hz] */
#define ALGO_FREQ_ENV   2U
/* Update period for environmental sensor [ms] */
#define ALGO_PERIOD_ENV (1000U / ALGO_FREQ_ENV)
/* 10kHz/2  for environmental @2Hz */
#define DEFAULT_uhCCR1_Val      (10000U / ALGO_FREQ_ENV)
    
/* Update frequency for mic audio level [Hz] */
#define ALGO_FREQ_AUDIO_LEVEL   20U
/* Update period for mic audio level [ms] */
#define ALGO_PERIOD_AUDIO_LEVEL (1000U / ALGO_FREQ_AUDIO_LEVEL)
/* 10kHz/20  for mic audio level @20Hz */
#define DEFAULT_uhCCR2_Val      (10000U / ALGO_FREQ_AUDIO_LEVEL)
    
/* Update frequency for Acc/Gyro/Mag sensor [Hz] */
#define FREQ_ACC_GYRO_MAG              200U 
/* Update period for Acc/Gyro/Mag [ms] */
#define ALGO_PERIOD_ACC_GYRO_MAG        (1000U / FREQ_ACC_GYRO_MAG) 
/* 10kHz/20  for Acc/Gyro/Mag @20Hz */
#define DEFAULT_uhCCR3_Val              (10000U / FREQ_ACC_GYRO_MAG)

/* Update frequency for battery info [Hz] */
#define ALGO_FREQ_BATTERY_INFO   2U
/* Update period for environmental sensor [ms] */
#define ALGO_PERIOD_BATTERY_INFO (1000U / ALGO_FREQ_BATTERY_INFO)
/* 10kHz/2  for environmental @2Hz */
#define DEFAULT_uhCCR4_Val      (10000U / ALGO_FREQ_BATTERY_INFO)

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

/**
  * @}
  */


#endif /* __MAIN_H */

/******************* (C) COPYRIGHT 2021 STMicroelectronics *****END OF FILE****/
