/**
  ******************************************************************************
  * @file    BLE_Battery.h
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   Battery info services APIs.
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
#ifndef _BLE_BATTERY_H_
#define _BLE_BATTERY_H_

#ifdef __cplusplus
 extern "C" {
#endif

/* Exported Variables ------------------------------------------------------- */
extern BLE_NotifyEnv_t BLE_Battery_NotifyEvent;

/* Exported functions ------------------------------------------------------- */

/**
 * @brief  Init battery info service
 * @param  None
 * @retval BleCharTypeDef* BleCharPointer: Data structure pointer for battery info service
 */
extern BleCharTypeDef* BLE_InitBatteryService(void);

#ifndef BLE_MANAGER_SDKV2
/**
 * @brief  Setting Battery Advertise Data
 * @param  uint8_t *manuf_data: Advertise Data
 * @retval None
 */
extern void BLE_SetBatteryAdvertizeData(uint8_t *manuf_data);
#endif /* BLE_MANAGER_SDKV2 */

/**
 * @brief  Update Battery characteristic
 * @param  int32_t BatteryLevel %Charge level
 * @param  uint32_t Voltage Battery Voltage
 * @param  uint32_t Current Battery Current (0x8000 if not available)
 * @param  uint32_t Status Charging/Discharging
 * @retval tBleStatus   Status
 */
tBleStatus BLE_BatteryUpdate(uint32_t BatteryLevel, uint32_t Voltage, uint32_t Current, uint32_t Status);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_BATTERY_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
