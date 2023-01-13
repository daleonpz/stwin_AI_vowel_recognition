/**
  ******************************************************************************
  * @file    MotionSP_Manager_template.h
  * @author  System Research & Applications Team - Catania Lab
  * @version 2.3.2
  * @date    21-Oct-2020
  * @brief 	 MotionSP_Manager_template.c Header File
  * 		 This file should be copied to the application folder and renamed
  * 		 to MotionSP_Manager.h.
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
#ifndef __MOTIONSP_MANAGER_H
#define __MOTIONSP_MANAGER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "MotionSP.h"
#include "idp005_motion_sensors.h"
#include "MotionSP_Threshold.h"

/** @addtogroup Projects
  * @{
  */

/** @addtogroup DEMONSTRATIONS Demonstrations
  * @{
  */

/** @addtogroup PREDICTIVE_MAINTENANCE_SRV Predictive Maintenance SRV
  * @{
  */

/** @addtogroup STM32_MOTIONSP_MANAGER MotionSP Manager
  * @{
  */

/** @defgroup STM32_MOTIONSP_MANAGER_EXPORTED_DEFINES STM32 Motion Signal Processing Manager Exported Defines
  * @{
  */

#define ACCELERO_ODR_DEFAULT                IIS3DWB_XL_ODR_26k7Hz
#define ACCELERO_FS_DEFAULT                 4
#define ACCELERO_HW_FILTER_DEFAULT          IIS3DWB_HP_ODR_DIV_800
#define ACCELERO_FIFO_BDR_DEFAULT           IIS3DWB_XL_BATCHED_AT_26k7Hz
#define ACCELERO_FIFO_STREAM_MODE           IIS3DWB_STREAM_MODE
#define ACCELERO_FIFO_BYPASS_MODE           IIS3DWB_BYPASS_MODE
  
/* Use (1) or not (0) the SW filter  */
#define USE_SW_HP_FILTER                    0
    
/* Limit (1) or not (0) FFT result within sensor bandwidth  */
#define LIMIT_FFT_RESULT                    0
  
#define ACCELEROMETER_PARAMETER_NRMAX       4
#define MOTIONSP_PARAMETER_NRMAX            7

/**
  * @}
  */  
  
/** @defgroup STM32_MOTIONSP_MANAGER_EXPORTED_MACROS STM32 Motion Signal Processing Manager Exported Macros
  * @{
  */

#define MAX(_A_, _B_)              (Alarm_Type_t)(_A_ > _B_ ? _A_ : _B_)
#define MAX4(_A_, _B_ ,_C_ ,_D_)   (Alarm_Type_t)(MAX(_A_, _B_) > MAX(_C_, _D_) ? MAX(_A_, _B_) : MAX(_C_, _D_)) 

/**
  * @}
  */  
  
/** @defgroup STM32_MOTIONSP_MANAGER_EXPORTED_TYPES STM32 Motion Signal Processing Manager Exported Types
  * @{
  */

/**
 * @brief  Struct for Accelerometer parameters
 */
typedef struct
{
  uint16_t AccOdr;          //!< Accelerometer ODR nominal value in Hz
  uint8_t AccOdrCode;       //!< Accelerometer driver ODR code
  uint16_t AccFifoBdr;      //!< Accelerometer FIFO BDR value in Hz
  uint8_t AccFifoBdrCode;   //!< Accelerometer driver FIFO BDR code
  uint16_t fs;              //!< Accelerometer full scale in g
  uint8_t HwFilter;         //!< Accelerometer HW filter
  float AccSens;            //!< Accelerometer sensitivity in [mg/LSB]
  uint16_t FifoWtm;         //!< Accelerometer FIFO watermark
} sAccPmtr_t;

/**
  * @}
  */  

/** @defgroup STM32_MOTIONSP_MANAGER_EXPORTED_FUNCTION PROTOTYPE Exported Function Prototypes
  * @{
  */

int32_t MotionSP_Init(void);
void MotionSP_AlarmThreshold_Init(void);
void MotionSP_TD_Threshold_Updating(uint8_t dflt);
void MotionSP_FD_Threshold_Updating(uint8_t subrange_num, uint8_t dflt);
int32_t MotionSP_MainManager(void);
void MotionSP_DataReady_IRQ_Rtn(void);
void MotionSP_FifoFull_IRQ_Rtn(void);

void MotionSP_TimeDomainAlarmInit (sTimeDomainAlarm_t *pTdAlarm,
                                   sTimeDomainData_t *pTimeDomainVal,
                                   sTimeDomainThresh_t *pTdRmsThreshold,
                                   sTimeDomainThresh_t *pTdPkThreshold);

void MotionSP_TimeDomainAlarm (sTimeDomainAlarm_t *pTdAlarm,
                               sTimeDomainData_t *pTimeDomainVal,
                               sTimeDomainThresh_t *pTdRmsThreshold,
                               sTimeDomainThresh_t *pTdPkThreshold,
                               sTimeDomainData_t *pTimeDomain);


int32_t MotionSP_FreqDomainAlarmInit (float **pWarnThresh,
                                      float **pAlarmThresh,
                                      sFreqDomainAlarm_t *pTHR_Fft_Alarms,
                                      uint8_t subrange_num);

void MotionSP_FreqDomainAlarm (sSubrange_t *pSRAmplitude,
                               float *pFDWarnThresh,
                               float *pFDAlarmThresh,
                               uint8_t subrange_num, 
                               sSubrange_t *pTHR_Check, 
                               sFreqDomainAlarm_t *pTHR_Fft_Alarms);

void MotionSP_TotalStatusAlarm(sTimeDomainAlarm_t *pTdAlarm,
                                      sFreqDomainAlarm_t *pTHR_Fft_Alarms,
                                      uint8_t subrange_num,
                                      Alarm_Type_t *pTotalTDAlarm,
                                      Alarm_Type_t *pTotalFDAlarm);

/** @defgroup STM32_MOTIONSP_MANAGER_EXPORTED_VARIABLES STM32 Motion Signal Processing Manager Exported Variables
  * @{
  */

extern sAccPmtr_t AcceleroParams;

extern sTimeDomainAlarm_t sTdAlarm;
extern sTimeDomainThresh_t sTdRmsThresholds;
extern sTimeDomainThresh_t sTdPkThresholds;
extern sTimeDomainData_t sTimeDomainVal;
extern sSubrange_t THR_Check;
extern sFreqDomainAlarm_t THR_Fft_Alarms;
extern float *pFDAlarmThresh;
extern float *pFDWarnThresh;
extern Alarm_Type_t TotalTDAlarm;
extern Alarm_Type_t TotalFDAlarm;

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __MOTIONSP_MANAGER_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
