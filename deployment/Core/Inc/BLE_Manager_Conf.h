/**
  ******************************************************************************
  * @file    BLE_Manager_Conf.h
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
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
  * SLA0055, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0055
  *
  ******************************************************************************
 */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __BLE_MANAGER_CONF_H__
#define __BLE_MANAGER_CONF_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Exported define ------------------------------------------------------------*/

/* Define the Max dimesion of the Bluetooth characteristics for each packet  */
#define DEFAULT_MAX_CHAR_LEN 155
   
#define BLE_MANAGER_MAX_ALLOCABLE_CHARS 32
  
/* For enabling the capability to handle BlueNRG Congestion */
#define ACC_BLUENRG_CONGESTION

/* Define the Delay function to use inside the BLE Manager (HAL_Delay/osDelay) */
#define BLE_MANAGER_DELAY HAL_Delay
  
/****************** Malloc/Free **************************/
#define BLE_MallocFunction malloc
#define BLE_FreeFunction free
  
/*---------- Print messages from BLE Manager files at middleware level -----------*/
/* Uncomment/Comment the following define for  disabling/enabling print messages from BLE Manager files */
#define BLE_MANAGER_DEBUG
  
#define BLE_DEBUG_LEVEL 1

#ifdef BLE_MANAGER_DEBUG
  /**
  * User can change here printf with a custom implementation.
  * For example:
  * #include "STBOX1_config.h"
  * #include "main.h"
  * #define BLE_MANAGER_PRINTF	STBOX1_PRINTF
  */

  #include "PREDMNT1_config.h"
  #include "main.h"
  #define BLE_MANAGER_PRINTF	PREDMNT1_PRINTF

//  #include <stdio.h>
//  #define BLE_MANAGER_PRINTF(...)	printf(__VA_ARGS__)
#else
  #define BLE_MANAGER_PRINTF(...)
#endif

#ifdef __cplusplus
}
#endif

#endif /* __BLE_MANAGER_CONF_H__*/

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
