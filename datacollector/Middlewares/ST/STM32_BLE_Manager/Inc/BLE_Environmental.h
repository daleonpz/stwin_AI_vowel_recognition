/**
  ******************************************************************************
  * @file    BLE_Environmental.h 
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   Environmental info services APIs.
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
#ifndef _BLE_ENVIRONMENTAL_H_
#define _BLE_ENVIRONMENTAL_H_

#ifdef __cplusplus
 extern "C" {
#endif
   
/* Exported typedef --------------------------------------------------------- */
typedef void (*CustomReadRequestEnv_t)(void);

/* Exported Variables ------------------------------------------------------- */
extern BLE_NotifyEnv_t BLE_Env_NotifyEvent;
extern CustomReadRequestEnv_t CustomReadRequestEnv;

/* Exported functions ------------------------------------------------------- */

/**
 * @brief  Init environmental info service
 * @param  uint8_t PressEnable:    1 for enabling the BLE pressure feature, 0 otherwise.
 * @param  uint8_t HumEnable:      1 for enabling the BLE humidity feature, 0 otherwise.
 * @param  uint8_t NumTempEnabled: 0 for disabling BLE temperature feature
 *                                 1 for enabling only one BLE temperature feature
 *                                 2 for enabling two BLE temperatures feature
 * @retval BleCharTypeDef* BleCharPointer: Data structure pointer for environmental info service
 */
extern BleCharTypeDef* BLE_InitEnvService(uint8_t PressEnable, uint8_t HumEnable, uint8_t NumTempEnabled);

#ifndef BLE_MANAGER_SDKV2
/**
 * @brief  Setting Environmental Advertise Data
 * @param  uint8_t *manuf_data: Advertise Data
 * @retval None
 */
extern void BLE_SetEnvAdvertizeData(uint8_t *manuf_data);
#endif /* BLE_MANAGER_SDKV2 */

/**
 * @brief  Update Environmental characteristic value
 * @param  int32_t Press:       Pressure in mbar (Set 0 if not used)
 * @param  uint16_t Hum:        humidity RH (Relative Humidity) in thenths of % (Set 0 if not used)
 * @param  int16_t Temp1:       Temperature in tenths of degree (Set 0 if not used)
 * @param  int16_t Temp2:       Temperature in tenths of degree (Set 0 if not used)
 * @retval tBleStatus:          Status
 */
extern tBleStatus BLE_EnvironmentalUpdate(int32_t Press, uint16_t Hum, int16_t Temp1, int16_t Temp2);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_ENVIRONMENTAL_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
