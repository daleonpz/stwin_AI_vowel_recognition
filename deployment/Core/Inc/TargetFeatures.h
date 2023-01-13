/**
  ******************************************************************************
  * @file    TargetFeatures.h 
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   Specification of the HW Features for each target platform
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
#ifndef _TARGET_FEATURES_H_
#define _TARGET_FEATURES_H_

#ifdef __cplusplus
 extern "C" {
#endif 

/* Includes ------------------------------------------------------------------*/
#include <stdlib.h>
   
#include "stm32l4xx_hal.h"
#include "stm32l4xx_hal_conf.h"

#include "STWIN.h"
#include "STWIN_audio.h"
#include "STWIN_motion_sensors_Patch.h"
#include "STWIN_motion_sensors_ex_Patch.h"
#include "STWIN_env_sensors.h"
#include "STWIN_bc.h"
   
#include "PREDMNT1_config.h"
#include "MetaDataManager.h"
   
/** @addtogroup Projects
  * @{
  */

/** @addtogroup DEMONSTRATIONS Demonstrations
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE Predictive Maintenance BLE
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM Predictive Maintenance Target Platform
  * @{
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_EXPORTED_DEFINES Predictive Maintenance Target Platform Exported Defines
  * @{
  */
   
/* Exported defines ------------------------------------------------------- */
#define MAX_TEMP_SENSORS 2
   
/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_EXPORTED_TYPES Predictive Maintenance Target Platform Exported Types
  * @{
  */

/* Exported types ------------------------------------------------------- */
   
/**
 * @brief  Target's Features data structure definition
 */
typedef struct
{
  uint8_t TempSensorsIsInit[MAX_TEMP_SENSORS];
  uint8_t PressSensorIsInit;
  uint8_t HumSensorIsInit;

  uint8_t AccSensorIsInit;
  uint8_t GyroSensorIsInit;
  uint8_t MagSensorIsInit;  
  
  int32_t NumTempSensors;
  int32_t NumMicSensors;

  uint8_t LedStatus;
} TargetFeatures_t;

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_EXPORTED_VARIABLES Predictive Maintenance Target Platform Exported Variables
  * @{
  */

/* Exported variables ------------------------------------------------------- */
extern TargetFeatures_t TargetBoardFeatures;

extern uint16_t PCM_Buffer[];

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_EXPORTED_FUNCTIONS_PROTOTYPES Predictive Maintenance Target Platform Exported Functions Prototypes
  * @{
  */

/* Exported functions ------------------------------------------------------- */
extern void InitTargetPlatform(void);

extern void InitMics(uint32_t AudioFreq, uint32_t AudioVolume);
extern void DeInitMics(void);

extern void LedOnTargetPlatform(void);
extern void LedOffTargetPlatform(void);
extern void LedToggleTargetPlatform(void);

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

#endif /* _TARGET_FEATURES_H_ */

/******************* (C) COPYRIGHT 2021 STMicroelectronics *****END OF FILE****/

