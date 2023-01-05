/**
  ******************************************************************************
  * @file    MetaDataManager_Config_Template.h 
  * @author  System Research & Applications Team - Catania Lab.
  * @version V1.3.0
  * @date    01-Dic-2020
  * @brief   Meta Data Manager Config Header File
  *          This file should be copied to the application folder and renamed
  *          to MetaDataManager_Config.h.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
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
#ifndef _META_DATA_MANAGER_CONFIG_H_
#define _META_DATA_MANAGER_CONFIG_H_

#ifdef __cplusplus
 extern "C" {
#endif 

/* Includes ---------------------------------------------------- */
#include <stdlib.h>

/* Replace the header file names with the ones of the target platform */
#include "stm32yyxx_hal.h"
#include "stm32yyxx_nucleo.h"

/*---------- Print messages from MetaDataManager files at middleware level -----------*/
#define MDM_DEBUG	0

#if MDM_DEBUG
	/**
	* User can change here printf with a custom implementation.
	* For example:
	* #include "PREDMNT1_config.h"
	* #define MDM_PRINTF	PREDMNT1_PRINTF
	*/
	#include <stdio.h>
	#define MDM_PRINTF(...)	printf(__VA_ARGS__)
#else
	#define MDM_PRINTF(...)
#endif

#ifdef __cplusplus
}
#endif

#endif /* _META_DATA_MANAGER_CONFIG_H_ */

/******************* (C) COPYRIGHT 2020 STMicroelectronics *****END OF FILE****/

