/**
  ******************************************************************************
  * @file    BLE_Inertial.h 
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   Inertial info services APIs.
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
#ifndef _BLE_INERTIAL_H_
#define _BLE_INERTIAL_H_

#ifdef __cplusplus
 extern "C" {
#endif

/* Exported Variables ------------------------------------------------------- */
extern BLE_NotifyEnv_t BLE_Inertial_NotifyEvent;

/* Exported Types ------------------------------------------------------- */
typedef struct
{
  int32_t x;
  int32_t y;
  int32_t z;
} BLE_MANAGER_INERTIAL_Axes_t;

/* Exported functions ------------------------------------------------------- */

/**
 * @brief  Init inertial info service
 * @param  uint8_t AccEnable:   1 for enabling the BLE accelerometer feature, 0 otherwise.
 * @param  uint8_t GyroEnable:  1 for enabling the BLE gyroscope feature, 0 otherwise.
 * @param  uint8_t MagEnabled:  1 for esabling the BLE magnetometer feature, 0 otherwise.
 * @retval BleCharTypeDef* BleCharPointer: Data structure pointer for environmental info service
 */
extern BleCharTypeDef* BLE_InitInertialService(uint8_t AccEnable, uint8_t GyroEnable, uint8_t MagEnabled);

#ifndef BLE_MANAGER_SDKV2
/**
 * @brief  Setting Inertial Advertise Data
 * @param  uint8_t *manuf_data: Advertise Data
 * @retval None
 */
extern void BLE_SetInertialAdvertizeData(uint8_t *manuf_data);
#endif /* BLE_MANAGER_SDKV2 */

/**
 * @brief  Update acceleration/Gryoscope and Magneto characteristics value
 * @param  BLE_MANAGER_INERTIAL_Axes_t Acc:     Structure containing acceleration value in mg
 * @param  BLE_MANAGER_INERTIAL_Axes_t Gyro:    Structure containing Gyroscope value
 * @param  BLE_MANAGER_INERTIAL_Axes_t Mag:     Structure containing magneto value
 * @retval tBleStatus      Status
 */
extern tBleStatus BLE_AccGyroMagUpdate(BLE_MANAGER_INERTIAL_Axes_t *Acc,
                                       BLE_MANAGER_INERTIAL_Axes_t *Gyro,
                                       BLE_MANAGER_INERTIAL_Axes_t *Mag);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_INERTIAL_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
