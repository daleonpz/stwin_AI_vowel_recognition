/**
  ******************************************************************************
  * @file    BLE_Implementation.h 
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   BLE Implementation header template file.
  *          This file should be copied to the application folder and renamed
  *          to BLE_Implementation.h.
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
#ifndef _BLE_IMPLEMENTATION_H_
#define _BLE_IMPLEMENTATION_H_

#ifdef __cplusplus
 extern "C" {
#endif 

/* Includes ------------------------------------------------------------------*/

/**
* User can added here the header file for the selected BLE features.
* For example:
* #include "BLE_Environmental.h"
* #include "BLE_Inertial.h"
*/

#include "BLE_Environmental.h"
#include "BLE_Inertial.h"
#include "BLE_AudioLevel.h"
#include "BLE_Battery.h"
#include "BLE_FFT_Amplitude.h"
#include "BLE_TimeDomain.h"
#include "BLE_FFT_AlarmSpeedStatus.h"
#include "BLE_FFT_AlarmAccPeakStatus.h"
#include "BLE_FFT_AlarmSubrangeStatus.h"

/* Exported Defines --------------------------------------------------------*/
     
/* Select the used hardware platform
 *
 * STEVAL-WESU1                         --> BLE_MANAGER_STEVAL_WESU1_PLATFORM
 * STEVAL-STLKT01V1 (SensorTile)        --> BLE_MANAGER_SENSOR_TILE_PLATFORM
 * STEVAL-BCNKT01V1 (BlueCoin)          --> BLE_MANAGER_BLUE_COIN_PLATFORM
 * STEVAL-IDB008Vx                      --> BLE_MANAGER_STEVAL_IDB008VX_PLATFORM
 * STEVAL-BCN002V1B (BlueTile)          --> BLE_MANAGER_STEVAL_BCN002V1_PLATFORM
 * STEVAL-MKSBOX1V1 (SensorTile.box)    --> BLE_MANAGER_SENSOR_TILE_BOX_PLATFORM
 * DISCOVERY-IOT01A                     --> BLE_MANAGER_DISCOVERY_IOT01A_PLATFORM
 * STEVAL-STWINKT1                      --> BLE_MANAGER_STEVAL_STWINKIT1_PLATFORM
 * STM32NUCLEO Board                    --> BLE_MANAGER_NUCLEO_PLATFORM
 *
 * For example:
 * #define BLE_MANAGER_USED_PLATFORM	BLE_MANAGER_NUCLEO_PLATFORM
 *
*/

/* Used platform */
#define BLE_MANAGER_USED_PLATFORM       BLE_MANAGER_STEVAL_STWINKIT1_PLATFORM

/* STM32 Unique ID */
#define BLE_STM32_UUID STM32_UUID

/* STM32  Microcontrolles type */
#define BLE_STM32_MICRO "L4R9"

/* STM32 board type*/
#define BLE_STM32_BOARD "STM32L4R9ZI-STWIN"

/* Package Version firmware */
#define BLE_VERSION_FW_MAJOR    PREDMNT1_VERSION_MAJOR
#define BLE_VERSION_FW_MINOR    PREDMNT1_VERSION_MINOR
#define BLE_VERSION_FW_PATCH    PREDMNT1_VERSION_PATCH

/* Firmware Package Name */
#define BLE_FW_PACKAGENAME      PREDMNT1_PACKAGENAME
   
/* Feature mask for Temperature1 */
#define FEATURE_MASK_TEMP1 0x00040000

/* Feature mask for Temperature2 */
#define FEATURE_MASK_TEMP2 0x00010000

/* Feature mask for Pressure */
#define FEATURE_MASK_PRESS 0x00100000

/* Feature mask for Humidity */
#define FEATURE_MASK_HUM   0x00080000

/* Feature mask for Accelerometer */
#define FEATURE_MASK_ACC   0x00800000

/* Feature mask for Gyroscope */
#define FEATURE_MASK_GRYO  0x00400000

/* Feature mask for Magnetometer */
#define FEATURE_MASK_MAG   0x00200000

/* Feature mask for Microphone */
#define FEATURE_MASK_MIC   0x04000000
   
/* Exported Variables ------------------------------------------------------- */
extern uint8_t connected;
extern int32_t  NeedToClearSecureDB;

/* Exported functions ------------------------------------------------------- */
extern void BLE_InitCustomService(void);
extern void BLE_SetCustomAdvertizeData(uint8_t *manuf_data);
extern void BluetoothInit(void);

#ifdef __cplusplus
}
#endif

#endif /* _BLE_IMPLEMENTATION_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
