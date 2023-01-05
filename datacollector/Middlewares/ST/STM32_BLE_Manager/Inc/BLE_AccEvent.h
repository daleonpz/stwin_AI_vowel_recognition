/**
  ******************************************************************************
  * @file    BLE_AccEvent.h 
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   Acceleromenter HW Event service APIs.
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
#ifndef _BLE_ACC_EVENT_H_
#define _BLE_ACC_EVENT_H_

#ifdef __cplusplus
 extern "C" {
#endif
   
/* Exported typedef --------------------------------------------------------- */
typedef void (*CustomReadRequestAccEvent_t)(void);

/* Exported Variables ------------------------------------------------------- */
extern BLE_NotifyEnv_t BLE_AccEnv_NotifyEvent;
extern CustomReadRequestAccEvent_t CustomReadRequestAccEvent;

/* Exported functions ------------------------------------------------------- */

/**
 * @brief  Init HW Acceleromenter Event info service
 * @param  None
 * @retval BleCharTypeDef* BleCharPointer: Data structure pointer for HW Acceleromenter Event info service
 */
extern BleCharTypeDef* BLE_InitAccEnvService(void);

#ifndef BLE_MANAGER_SDKV2
/**
 * @brief  Setting HW Acceleromenter Event Advertise Data
 * @param  uint8_t *manuf_data: Advertise Data
 * @retval None
 */
extern void BLE_SetAccEnvAdvertizeData(uint8_t *manuf_data);
#endif /* BLE_MANAGER_SDKV2 */

/**
 * @brief  Update HW Acceleromenter Event characteristic value
 * @param  uint16_t Command to Send
 * @param  uint8_t Command Lenght 
 * @retval tBleStatus: Status
 */
extern tBleStatus BLE_AccEnvUpdate(uint16_t Command, uint8_t dimByte);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_ACC_EVENT_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
