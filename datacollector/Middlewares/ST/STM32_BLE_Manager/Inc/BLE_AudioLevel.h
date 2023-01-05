/**
  ******************************************************************************
  * @file    BLE_AudioLevel.h 
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   Audio level info services APIs.
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
#ifndef _BLE_AUDIO_LEVEL_H_
#define _BLE_AUDIO_LEVEL_H_

#ifdef __cplusplus
 extern "C" {
#endif 

/* Exported Variables ------------------------------------------------------- */
extern BLE_NotifyEnv_t BLE_AudioLevel_NotifyEvent;

/* Exported functions ------------------------------------------------------- */

/**
 * @brief  Init audio level info service
 * @param  uint8_t AudioLevelNumber: Number of audio level features (up to 4 audio level are supported) 
 * @retval BleCharTypeDef* BleCharPointer: Data structure pointer for audio level info service
 */
extern BleCharTypeDef* BLE_InitAudioLevelService(uint8_t AudioLevelNumber);

#ifndef BLE_MANAGER_SDKV2
/**
 * @brief  Setting Environmental Advertise Data
 * @param  uint8_t *manuf_data: Advertise Data
 * @retval None
 */
extern void BLE_SetAudioLevelAdvertizeData(uint8_t *manuf_data);
#endif /* BLE_MANAGER_SDKV2 */

/**
 * @brief  Update audio level characteristic values
 * @param  uint16_t *AudioLevelData:    SNR dB audio level array
 * @param  uint8_t AudioLevelNumber:    Number of audio level features (up to 4 audio level are supported)
 * @retval tBleStatus   Status
 */
tBleStatus BLE_AudioLevelUpdate(uint16_t *AudioLevelData, uint8_t AudioLevelNumber);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_AUDIO_LEVEL_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
