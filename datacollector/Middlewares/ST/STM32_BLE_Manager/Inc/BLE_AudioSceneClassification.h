/**
  ******************************************************************************
  * @file    BLE_AudioSceneClasssification.h
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   Audio Scene Classification info service APIs.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0094, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0094
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/  
#ifndef _BLE_AUDIO_SCENE_CALSSIFICATION_H_
#define _BLE_AUDIO_SCENE_CALSSIFICATION_H_

#ifdef __cplusplus
 extern "C" {
#endif

/* Exported defines ---------------------------------------------------------*/

/* Exported typedef --------------------------------------------------------- */
typedef void (*CustomReadRequestAudioSceneClass_t)(void);

typedef enum
{
  BLE_ASC_HOME      = 0x00,
  BLE_ASC_OUTDOOR   = 0x01,
  BLE_ASC_CAR       = 0x02,
  BLE_ASC_OFF       = 0xF0, //Off
  BLE_ASC_ON        = 0xF1, //On
  BLE_ASC_UNDEFINED = 0xFF
} BLE_ASC_output_t;

/* Exported Variables ------------------------------------------------------- */
extern BLE_NotifyEnv_t BLE_AudioSceneClass_NotifyEvent;
extern CustomReadRequestAudioSceneClass_t CustomReadRequestAudioSceneClass;

/* Exported functions ------------------------------------------------------- */

/**
 * @brief  Init Audio Scene Classification info service
 * @param  None
 * @retval BleCharTypeDef* BleCharPointer: Data structure pointer for Activity Classification info service
 */
extern BleCharTypeDef* BLE_InitAudioSceneClassService(void);

/**
 * @brief  Update Audio Scene Classification characteristic
 * @param  BLE_ASC_output_t ASC_Code Audio Scene Classification Code
 * @retval tBleStatus   Status
 */
extern tBleStatus BLE_AudioSceneClassUpdate(BLE_ASC_output_t ASC_Code);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_AUDIO_SCENE_CALSSIFICATION_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
