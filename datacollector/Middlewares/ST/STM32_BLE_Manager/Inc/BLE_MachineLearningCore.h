/**
  ******************************************************************************
  * @file    BLE_MachineLearningCore.h
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   Machine Learning Core info services APIs.
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
#ifndef _BLE_MACHINE_LEARNING_CORE_H_
#define _BLE_MACHINE_LEARNING_CORE_H_

#ifdef __cplusplus
 extern "C" {
#endif 

/* Exported Variables ------------------------------------------------------- */
extern BLE_NotifyEnv_t BLE_MachineLearningCore_NotifyEvent;

/* Exported functions ------------------------------------------------------- */

/**
 * @brief  Init Machine Learning Core info service
 * @param  None
 * @retval BleCharTypeDef* BleCharPointer: Data structure pointer for Machine Learning Core info service
 */
BleCharTypeDef* BLE_InitMachineLearningCoreService(void);

/**
 * @brief  Setting Machine Learning Core Advertize Data
 * @param  uint8_t *manuf_data: Advertize Data
 * @retval None
 */
void BLE_SetMachineLearningCoreAdvertizeData(uint8_t *manuf_data);

/**
 * @brief  Update Machine Learning Core output registers characteristic
 * @param  uint8_t *mlc_out				pointers to 8 MLC register
 * @param  uint8_t *mlc_status_mainpage	pointer to the MLC status bits from 1 to 8
 * @retval tBleStatus Status
 */
tBleStatus BLE_MachineLearningCoreUpdate(uint8_t *mlc_out, uint8_t *mlc_status_mainpage);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_MACHINE_LEARNING_CORE_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
