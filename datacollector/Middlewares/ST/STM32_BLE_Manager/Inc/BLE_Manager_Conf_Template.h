/**
  ******************************************************************************
  * @file    BLE_Manager_Conf_Template.h
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   BLE Manager configuration template file.
  *          This file should be copied to the application folder and renamed
  *          to BLE_Manager_Conf.h.
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
#ifndef __BLE_MANAGER_CONF_H__
#define __BLE_MANAGER_CONF_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Exported define -----------------------------------------------------------*/
  
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
 * Undefined platform                   --> BLE_MANAGER_UNDEF_PLATFORM
*/
  
/* Identify the used hardware platform  */
#define BLE_MANAGER_USED_PLATFORM	BLE_MANAGER_UNDEF_PLATFORM

/* Define the Max dimesion of the Bluetooth characteristics for each packet */
#define DEFAULT_MAX_CHAR_LEN 155
   
#define BLE_MANAGER_MAX_ALLOCABLE_CHARS 32

/* For enabling the capability to handle BlueNRG Congestion */
#define ACC_BLUENRG_CONGESTION
  
/* Define the Delay function to use inside the BLE Manager */
#define BLE_MANAGER_DELAY HAL_Delay
  
/****************** Malloc/Free **************************/
#define BLE_MallocFunction malloc
#define BLE_FreeFunction free
  
/*---------- Print messages from BLE Manager files at middleware level -------*/
/* Uncomment the following define for  enabling print debug messages */
#define BLE_MANAGER_DEBUG

#ifdef BLE_MANAGER_DEBUG
  /**
  * User can change here printf with a custom implementation.
  */

 #include <stdio.h>
 #define BLE_MANAGER_PRINTF(...)	printf(__VA_ARGS__)
 
 /* Define the Debug Level: 1/2/3(default value) */
 #define BLE_DEBUG_LEVEL 1
#else
  #define BLE_MANAGER_PRINTF(...)
#endif

/*---------------- Don't change the following defines ------------------------*/
//#define BLE_MANAGER_SDKV2

#ifdef __cplusplus
}
#endif

#endif /* __BLE_MANAGER_CONF_H__*/

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
