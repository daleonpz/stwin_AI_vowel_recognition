/**
  ******************************************************************************
  * @file    MotionSP_Manager.h
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   Header for MotionSP_Manager.c
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
#ifndef __MOTIONSP_MANAGER_H
#define __MOTIONSP_MANAGER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "MotionSP.h"
#include "MotionSP_Threshold.h"
#include "TargetFeatures.h"

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

/* Set defaul ODR value to 6600Hz for FFT demo */
#define ACCELERO_ODR_DEFAULT            6660

/* Default value for Accelerometer full scale in g */
#define ACCELERO_FS_DEFAULT             4
  
#define ACCELERO_HW_FILTER_DEFAULT      HPF_ODR_DIV_400
  
/* Set defaul FIFO ODR value */
#define ACCELERO_FIFO_BDR_DEFAULT       ACCELERO_ODR_DEFAULT
                                        
#define ACCELERO_FIFO_STREAM_MODE       ISM330DHCX_STREAM_MODE
#define ACCELERO_FIFO_BYPASS_MODE       ISM330DHCX_BYPASS_MODE
#define ACCELERO_DRDY_PULSED            ISM330DHCX_DRDY_PULSED

/* Complete size for FIFO frame for ism330dhcx */
#define FIFO_FRAME_SIZE 7
/* Fifo dimension in bytes */
#define MAX_FIFO_SIZE (1024 * 3)
  
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

#define ALARM_MAX(_A_, _B_)             (Alarm_Type_t)(_A_ > _B_ ? _A_ : _B_)
#define MAX4(_A_, _B_ ,_C_ ,_D_)        (Alarm_Type_t)(ALARM_MAX(_A_, _B_) > ALARM_MAX(_C_, _D_) ? ALARM_MAX(_A_, _B_) : ALARM_MAX(_C_, _D_)) 

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

uint8_t EnableDisable_ACC_HP_Filter(uint8_t FilterIsEnabled);

int32_t enable_FIFO(void);
int32_t disable_FIFO(void);
int32_t MotionSP_AcceleroConfig(void);

void MotionSP_SetDefaultVibrationParam(void);


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
//extern Alarm_Type_t TotalTDAlarm;
//extern Alarm_Type_t TotalFDAlarm;

extern uint8_t MotionSP_Running;

extern volatile uint32_t FFT_Amplitude;
extern volatile uint32_t FFT_Alarm;

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
