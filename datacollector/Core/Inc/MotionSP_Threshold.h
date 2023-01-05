 /**
  ******************************************************************************
  * @file    MotionSP_threshold.h
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
#ifndef _MOTIONSP_TH_H_
#define _MOTIONSP_TH_H_

#ifdef __cplusplus
extern "C" {
#endif
 
#include "MotionSP.h"

/** @addtogroup Projects
  * @{
  */

/** @addtogroup DEMONSTRATIONS Demonstrations
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE Predictive Maintenance BLE
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE_MOTIONSP_MANAGER Predictive Maintenance Motion Signal Processing Manager
  * @{
  */

/** @addgroup PREDICTIVE_MAINTENANCE_MOTIONSP_MANAGER_THRESHOLD Predictive Maintenance Motion Signal Processing Manager Threshold
  * @verbatim

    LEGENDA FOR VIBRATION ANALYSIS ZONE
    -----------------------------------
    Zone A: The vibration of newly commissioned machines
            normally falls within this zone.
    Zone B: Machines with vibration within this zone are
            normally considered acceptable for
            unrestricted longterm operation.
    Zone C: Machines with vibration within this zone are
            normally considered unsatisfactory for
            long-term continuous operation. WARNING
    Zone D: Vibration values within this zone are
            normally considered to be of sufficient
            severity to cause damage to the machine.

  * @endverbatim 
  * @{
  */
  
/** @defgroup PREDICTIVE_MAINTENANCE_MOTIONSP_MANAGER_THRESHOLD_EXPORTED_TYPES Predictive Maintenance Motion Signal Processing Manager Threshold Exported Types
  * @{
  */

/**
 * @brief  Arrays for WARNING and ALARM on Speed RMS Time Domain 
 */
typedef struct
{
  float THR_WARN_AXIS_X;        //!< X WARNING Threshold for Time domain
  float THR_WARN_AXIS_Y;        //!< Y WARNING Threshold for Time domain
  float THR_WARN_AXIS_Z;        //!< Z WARNING Threshold for Time domain
  float THR_ALARM_AXIS_X;       //!< X ALARM Threshold for Time domain
  float THR_ALARM_AXIS_Y;       //!< X ALARM Threshold for Time domain
  float THR_ALARM_AXIS_Z;       //!< X ALARM Threshold for Time domain 
} sTimeDomainThresh_t;

/**
 * @brief  Warning Alarm Datatype
 */
typedef enum
{
  GOOD          = (uint8_t)0x00,  //!< GOOD STATUS for thresholds
  WARNING       = (uint8_t)0x01,  //!< WARNING STATUS for thresholds
  ALARM         = (uint8_t)0x02,  //!< ALARM STATUS for thresholds 
  NONE          = (uint8_t)0x03,  //!< RFU STATUS for thresholds      
} Alarm_Type_t;


/**
 * @brief  STATUS for Frequency domain Warning-Alarm
 */
typedef struct 
{
  Alarm_Type_t STATUS_AXIS_X[SUBRANGE_MAX]; //!< X STATUS ALARM for FreqDomain     
  Alarm_Type_t STATUS_AXIS_Y[SUBRANGE_MAX]; //!< Y STATUS ALARM for FreqDomain
  Alarm_Type_t STATUS_AXIS_Z[SUBRANGE_MAX]; //!< Z STATUS ALARM for FreqDomain
} sFreqDomainAlarm_t;


/**
 * @brief  STATUS for Time Domain Warning-Alarm
 */
typedef struct 
{
  Alarm_Type_t RMS_STATUS_AXIS_X;   //!< X STATUS ALARM for Speed RMS TimeDomain
  Alarm_Type_t RMS_STATUS_AXIS_Y;   //!< Y STATUS ALARM for Speed RMS TimeDomain
  Alarm_Type_t RMS_STATUS_AXIS_Z;   //!< Z STATUS ALARM for Speed RMS TimeDomain 
  Alarm_Type_t PK_STATUS_AXIS_X;    //!< X STATUS ALARM for Acc Peak TimeDomain
  Alarm_Type_t PK_STATUS_AXIS_Y;    //!< Y STATUS ALARM for Acc Peak TimeDomain
  Alarm_Type_t PK_STATUS_AXIS_Z;    //!< Z STATUS ALARM for Acc Peak TimeDomain 
} sTimeDomainAlarm_t;
  
/**
 * @brief  STATUS for bud subrange Warning-Alarm
 */
typedef struct 
{
  Alarm_Type_t STATUS_AXIS_X;    //!< X STATUS ALARM for Acc Peak TimeDomain
  Alarm_Type_t STATUS_AXIS_Y;    //!< Y STATUS ALARM for Acc Peak TimeDomain
  Alarm_Type_t STATUS_AXIS_Z;    //!< Z STATUS ALARM for Acc Peak TimeDomain 
} sBudSubrangeAlarm_t;

/**
  * @}
  */  

/** @defgroup PREDICTIVE_MAINTENANCE_MOTIONSP_MANAGER_THRESHOLD_EXPORTED_CONSTANTS Predictive Maintenance Motion Signal Processing Manager Threshold Exported Constants
  * @{
  */

/**
 * @brief Values inserted are just an example starting from information included in ISO10816-2/3
 * specs about steady state RMS Speed for Machinery rotating at RPM = 3000-3600.
 * In this cases we are choosing the medium values between the Zone Boundary A/B and B/C for the 
 * WARNING THRESHOLDS, and the medium values between the Zone Boundary B/C and C/D for the ALARM
 * "THE USER CAN CHANGE THESE VALUES TO ADAPT THE ANALYSYS WITH HIS SIGNATURE MACHINE OR CONDITIONS"
 */
static const sTimeDomainThresh_t TDSpeedRMSThresh = 
{/* Value in mm/s */
  5.65f,        //!< SPEED_RMS_THR_WARN_AXIS_X
  5.65f,        //!< SPEED_RMS_THR_WARN_AXIS_Y
  5.65f,        //!< SPEED_RMS_THR_WARN_AXIS_Z
  9.65f,        //!< SPEED_RMS_THR_ALARM_AXIS_X
  9.65f,        //!< SPEED_RMS_THR_ALARM_AXIS_Y
  9.65f,        //!< SPEED_RMS_THR_ALARM_AXIS_Z
}; 
 
/**
 * @brief Values inserted considering the value for Acceleration Peak for Machine at RPM=3000-3600
 *  and using an ideal shaker @60 Hz for the WARNING for the ALARM
 */
static const sTimeDomainThresh_t TDAccPeakThresh = 
{/* Value in m/s^2 */
  3.5f,         //!< THR_WARN_AXIS_X
  3.5f,         //!< THR_WARN_AXIS_Y
  3.5f,         //!< THR_WARN_AXIS_Z
  6.5f,         //!< THR_ALARM_AXIS_X
  6.5f,         //!< THR_ALARM_AXIS_Y
  6.5f,         //!< THR_ALARM_AXIS_Z
}; 

/****************************************************/
/*- WARNING and ALARM THRESHOLDS with SUBRANGE = 8 -*/
/****************************************************/

static const float FDWarnThresh_Sub8[8][3] = {
/*  -X-   -Y-   -Z- */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 1 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 2 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 3 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 4 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 5 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 6 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 7 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 8 */
};

static const float FDAlarmThresh_Sub8[8][3] = {
/*  -X-   -Y-   -Z- */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 1 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 2 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 3 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 4 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 5 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 6 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 7 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 8 */
};

/***************************************************/
/* WARNING and ALARM THRESHOLDS with SUBRANGE = 16 */
/***************************************************/
static const float FDWarnThresh_Sub16[16][3] = {
/*  -X-   -Y-   -Z- */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 1  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 2  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 3  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 4  */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 5  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 6  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 7  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 8  */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 9  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 10 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 11 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 12 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 13 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 14 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 15 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 16 */  
};

static const float FDAlarmThresh_Sub16[16][3] = {
/*  -X-   -Y-   -Z- */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 1  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 2  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 3  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 4  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 5  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 6  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 7  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 8  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 9  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 10 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 11 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 12 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 13 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 14 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 15 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 16 */  
};


/***************************************************/
/*-WARNING and ALARM THRESHOLDS with SUBRANGE = 32 */
/***************************************************/
static const float FDWarnThresh_Sub32[32][3] = {
/*  -X-   -Y-   -Z- */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 1  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 2  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 3  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 4  */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 5  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 6  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 7  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 8  */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 9  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 10 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 11 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 12 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 13 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 14 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 15 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 16 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 17 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 18 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 19 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 20 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 21 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 22 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 23 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 24 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 25 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 26 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 27 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 28 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 29 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 30 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 31 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 32 */
};

static const float FDAlarmThresh_Sub32[32][3] = {
/*  -X-   -Y-   -Z- */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 1  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 2  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 3  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 4  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 5  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 6  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 7  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 8  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 9  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 10 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 11 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 12 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 13 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 14 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 15 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 16 */  
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 17 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 18 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 19 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 20 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 21 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 22 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 23 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 24 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 25 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 26 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 27 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 28 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 29 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 30 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 31 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 32 */
};


/***************************************************/
/*-WARNING and ALARM THRESHOLDS with SUBRANGE = 64 */
/***************************************************/
static const float FDWarnThresh_Sub64[64][3] = {
/*  -X-   -Y-   -Z- */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 1  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 2  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 3  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 4  */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 5  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 6  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 7  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 8  */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 9  */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 10 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 11 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 12 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 13 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 14 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 15 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 16 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 17 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 18 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 19 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 20 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 21 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 22 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 23 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 24 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 25 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 26 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 27 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 28 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 29 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 30 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 31 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 32 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 33 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 34 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 35 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 36 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 37 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 38 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 39 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 40 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 41 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 42 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 43 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 44 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 45 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 46 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 47 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 48 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 49 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 50 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 51 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 52 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 53 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 54 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 55 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 56 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 57 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 58 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 59 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 60 */
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 61 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 62 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 63 */ 
  {1.5f, 2.5f, 3.5f},   /* Warn Thr Subrange 64 */
};

static const float FDAlarmThresh_Sub64[64][3] = {
/*  -X-   -Y-   -Z- */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 1  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 2  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 3  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 4  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 5  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 6  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 7  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 8  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 9  */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 10 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 11 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 12 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 13 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 14 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 15 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 16 */  
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 17 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 18 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 19 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 20 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 21 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 22 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 23 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 24 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 25 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 26 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 27 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 28 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 29 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 30 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 31 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 32 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 33 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 34 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 35 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 36 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 37 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 38 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 39 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 40 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 41 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 42 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 43 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 44 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 45 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 46 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 47 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 48 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 49 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 50 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 51 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 52 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 53 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 54 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 55 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 56 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 57 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 58 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 59 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 60 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 61 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 62 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 63 */
  {4.5f, 5.5f, 6.5f},   /* Alarm Thr Subrange 64 */
};

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

#endif //_MOTIONSP_TH_H_

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
