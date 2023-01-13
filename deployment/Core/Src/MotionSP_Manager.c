/**
  ******************************************************************************
  * @file    MotionSP_Manager.c
  * @author  System Research & Applications Team - Catania Lab
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   Application file to manage the Motion Signal Processing
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

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include "MotionSP_Manager.h"

/* malloc, free */
#include <stdlib.h> 


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

/** @defgroup STM32_MOTIONSP_MANAGER_PRIVATE_VARIABLES MotionSP Manager Private Variables
  * @{
  */
  
uint8_t RestartFlag = 1;
sAccPmtr_t AcceleroParams;
uint32_t StartTick = 0;
uint8_t MotionSP_Running= 0;

static uint16_t AccDrdyNr = 0;

uint8_t TD_Th_DFLT = 1;
uint8_t FD_Th_DFLT = 1;

/* X-Y-Z STATUS ALARM for TimeDomain */
sTimeDomainAlarm_t sTdAlarm;
/* X-Y-Z ALARM Threshold for Speed RMS TimeDomain */
sTimeDomainThresh_t sTdRmsThresholds;
/* X-Y-Z ALARM Threshold for Acc Peak TimeDomain */
sTimeDomainThresh_t sTdPkThresholds;
/* X-Y-Z parameter measured for TimeDomain */
sTimeDomainData_t sTimeDomainVal;
/* X-Y-Z Amplitude subranges Values that exceed thresholds */
sSubrange_t THR_Check;
/* X-Y-Z Frequency domain Subranges Status */
sFreqDomainAlarm_t THR_Fft_Alarms;

float *pFDAlarmThresh;
float *pFDWarnThresh;
//Alarm_Type_t TotalTDAlarm;
//Alarm_Type_t TotalFDAlarm;

float TDRmsThresh[3*2];
float TDPkThresh[3*2];
float FDWarnThresh_Sub[SUBRANGE_MAX*3];
float FDAlarmThresh_Sub[SUBRANGE_MAX*3];

/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/
static int32_t MotionSP_AccMeasInit(void);
static int32_t AccOdrMeas(sAcceleroODR_t *pAcceleroODR);
static void FillAccCircBuffFromFifo(sCircBuff_t *pAccCircBuff, uint8_t *pFifoBuff, uint16_t FifoWtm, float AccSens, uint8_t Rst);

/** @addtogroup STM32_MOTIONSP_MANAGER_PRIVATE_FUNCTIONS MotionSP Manager Private Functions
  * @{
  */

/**
  * @brief 	Measurement initialization for the accelerometer
  * @retval BSP status
  */    
static int32_t MotionSP_AccMeasInit(void)
{
  int32_t BSP_Error = BSP_ERROR_NONE;
  
  /** Evaluate the real accelerometer ODR *********************************/
  uint8_t iteration = 0;
  do
  {
    BSP_Error = AccOdrMeas(&AcceleroODR);
    iteration++;
  } while( (BSP_Error != BSP_ERROR_NONE) && (iteration < 3) );
  /************************************************************************/
  
  return BSP_Error;
}

/**
  * @brief 	Measurement of the accelerometer output data rate
  * @param 	pAcceleroODR Pointer to be fill with the new value
  * @retval BSP status
  */
static int32_t AccOdrMeas(sAcceleroODR_t *pAcceleroODR)
{
#define ODRMEASURINGTIME  1000  // in ms
#define TOLERANCE         5     // from 0 to 100
  
  int32_t BSP_Error = BSP_ERROR_NONE;
  uint32_t tkStart;
  uint32_t OdrRange[2] = {
    (AcceleroParams.AccOdr * (100-TOLERANCE))/100,
    (AcceleroParams.AccOdr * (100+TOLERANCE))/100
  };
  
  // The data-ready pulses are 75 �s long
  if ((BSP_Error = MOTION_SENSOR_DRDY_Set_Mode(ACCELERO_INSTANCE, ACCELERO_DRDY_PULSED)) != BSP_ERROR_NONE)
    return BSP_Error;
  
  /* IIS3DWB INT1_DRDY_XL interrupt enable */
  if ((BSP_Error = MOTION_SENSOR_Set_INT2_DRDY(ACCELERO_INSTANCE, ENABLE)) != BSP_ERROR_NONE)
    return BSP_Error;

  AccDrdyNr = 0;
  tkStart = BSP_GetTick();
  while ( (BSP_GetTick() - tkStart) < ODRMEASURINGTIME);
  
  /* IIS3DWB INT1_DRDY_XL interrupt disable */
  if ((BSP_Error = MOTION_SENSOR_Set_INT2_DRDY(ACCELERO_INSTANCE, DISABLE)) != BSP_ERROR_NONE)
    return BSP_Error;
  
  /* Calculate measured ODR and Exponential parameters*/
  pAcceleroODR->Frequency = (AccDrdyNr * 1000) / ODRMEASURINGTIME;
  
  if ( (pAcceleroODR->Frequency > OdrRange[0]) && (pAcceleroODR->Frequency < OdrRange[1]) )
  {
    /* ODR is valid */
    pAcceleroODR->Period = 1.0f/(float)pAcceleroODR->Frequency;
    pAcceleroODR->Tau= exp(-(float)(1000.0f*pAcceleroODR->Period)/(float)MotionSP_Parameters.tau);
  }
  else
    BSP_Error = BSP_ERROR_COMPONENT_FAILURE;
  
  return BSP_Error;
}

/**
  * @brief  Fills circular buffer from fifo of the accelerometer 
  * @retval None
  */
static void FillAccCircBuffFromFifo(sCircBuff_t *pAccCircBuff, uint8_t *pFifoBuff, uint16_t FifoWtm, float AccSens, uint8_t Rst)
{
  uint8_t *pFifoData = NULL;                  //!< FIFO pointer
  uint16_t i = 0;                             //!< FIFO index
  SensorVal_f_t mgAcc = {0.0, 0.0, 0.0};      //!< actual acceleration in mg
#if USE_SW_HP_FILTER == 1
  SensorVal_f_t mgAccNoDC = {0.0, 0.0, 0.0};  //!< actual acceleration in mg without DC component
#endif
  for(i=0; i<FifoWtm; i++)
  {
    pFifoData = pFifoBuff+(i*7);

    mgAcc.AXIS_X = (*(int16_t *)(pFifoData+1)) * AccSens;
    mgAcc.AXIS_Y = (*(int16_t *)(pFifoData+3)) * AccSens;
    mgAcc.AXIS_Z = (*(int16_t *)(pFifoData+5)) * AccSens;
    
#if USE_SW_HP_FILTER == 1
    // High Pass Filter to delete Accelerometer Offset
    MotionSP_accDelOffset(&mgAccNoDC, &mgAcc, DC_SMOOTH, Rst);
#endif
    
    /* Prepare the position to be filled */
    pAccCircBuff->IdPos += 1;
    
    /* Check for overflow */
    if (pAccCircBuff->IdPos == pAccCircBuff->Size)
    {
      pAccCircBuff->IdPos = 0;
      pAccCircBuff->Ovf = 1;
    }
    
#if USE_SW_HP_FILTER == 1
    pAccCircBuff->Array.X[pAccCircBuff->IdPos] = mgAccNoDC.AXIS_X * G_CONV;
    pAccCircBuff->Array.Y[pAccCircBuff->IdPos] = mgAccNoDC.AXIS_Y * G_CONV;
    pAccCircBuff->Array.Z[pAccCircBuff->IdPos] = mgAccNoDC.AXIS_Z * G_CONV;
#else
    pAccCircBuff->Array.X[pAccCircBuff->IdPos] = mgAcc.AXIS_X * G_CONV;
    pAccCircBuff->Array.Y[pAccCircBuff->IdPos] = mgAcc.AXIS_Y * G_CONV;
    pAccCircBuff->Array.Z[pAccCircBuff->IdPos] = mgAcc.AXIS_Z * G_CONV;
#endif
    
    if (Rst)
      Rst = 0;
  }
}
/**
  * @}
  */

/** @addtogroup STM32_MOTIONSP_MANAGER_EXPORTED_FUNCTIONS MotionSP_Manager Exported Functions
  * @{
  */

/**
  * @brief  Init the Accelerometer Settings and MotionSP Vibration parameters
  * @return None
  */   
void MotionSP_SetDefaultVibrationParam(void)
{
  memset((void *)&AcceleroParams, 0, sizeof(sAccPmtr_t));
  
  /* Set default parameters for accelerometer */
  AcceleroParams.AccOdr         = ACCELERO_ODR_DEFAULT;
  AcceleroParams.AccFifoBdr     = ACCELERO_FIFO_BDR_DEFAULT;
  AcceleroParams.fs             = ACCELERO_FS_DEFAULT;
  AcceleroParams.HwFilter       = ACCELERO_HW_FILTER_DEFAULT;
  
  /* Set default parameters for MotionSP library */
  MotionSP_Parameters.FftSize       = FFT_SIZE_DEFAULT;
  MotionSP_Parameters.tau           = TAU_DEFAULT;
  MotionSP_Parameters.window        = WINDOW_DEFAULT;
  MotionSP_Parameters.td_type       = TD_DEFAULT;
  MotionSP_Parameters.tacq          = TACQ_DEFAULT;
  MotionSP_Parameters.FftOvl        = FFT_OVL_DEFAULT;
  MotionSP_Parameters.subrange_num  = SUBRANGE_DEFAULT;
  
  PREDMNT1_PRINTF("\r\nVibration parameters have been set as default values\r\n");
}

/**
  * @brief  Set accelerometer parameters for MotionSP Vibration
  * @retval BSP status
  */   
int32_t MotionSP_AcceleroConfig(void)
{
  int32_t BSP_Error = BSP_ERROR_NONE;
//#if LIMIT_FFT_RESULT == 1
//  float LpfCutFreq = 0.0;
//  uint16_t MagSizeForLpf;
//#endif
  
  PREDMNT1_PRINTF("Accelero Config:\r\n");

  /* Set FS value */
  if ((BSP_Error =  MOTION_SENSOR_SetFullScale(ACCELERO_INSTANCE, MOTION_ACCELERO, AcceleroParams.fs)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError on FullScale Setting(BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk FullScale Setting\r\n");
  }
     
  /* Get Sensitivity */
  MOTION_SENSOR_GetSensitivity(ACCELERO_INSTANCE, MOTION_ACCELERO, &AcceleroParams.AccSens );
  
  /* Set ODR value */
  if ((BSP_Error = MOTION_SENSOR_SetOutputDataRate(ACCELERO_INSTANCE, MOTION_ACCELERO, AcceleroParams.AccOdr)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError Set Output Data Rate (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk Set Output Data Rate\r\n");
  }  
    
  /* Turn-on time delay */
  HAL_Delay(100);
  
  // Reset the real accelero ODR value
  memset((void *)&AcceleroODR, 0x00, sizeof(sAcceleroODR_t));
  
  /* Measure and calculate ODR */
  if ((BSP_Error = MotionSP_AccMeasInit()) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError measure and calculate ODR - Used parameter value (");
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk measure and calculate ODR (");
  }
  
  
//#if LIMIT_FFT_RESULT == 1
//  /* Calculate the number of FFT magnitute elements within Low Pass Filter range */
//  switch (AcceleroParams.HwFilter)
//  {
//  case IIS3DWB_LP_5kHz:
//    LpfCutFreq = 5000.0;
//    break;
//  case IIS3DWB_LP_ODR_DIV_4:
//    LpfCutFreq = AcceleroParams.AccOdr/4;
//    break;
//  case IIS3DWB_LP_ODR_DIV_10:
//    LpfCutFreq = AcceleroParams.AccOdr/10;
//    break;
//  case IIS3DWB_LP_ODR_DIV_20:
//    LpfCutFreq = AcceleroParams.AccOdr/20;
//    break;
//  case IIS3DWB_LP_ODR_DIV_45:
//    LpfCutFreq = AcceleroParams.AccOdr/45;
//    break;
//  case IIS3DWB_LP_ODR_DIV_100:
//    LpfCutFreq = AcceleroParams.AccOdr/100;
//    break;
//  case IIS3DWB_LP_ODR_DIV_200:
//    LpfCutFreq = AcceleroParams.AccOdr/200;
//    break;
//  case IIS3DWB_LP_ODR_DIV_400:
//    LpfCutFreq = AcceleroParams.AccOdr/400;
//    break;
//  case IIS3DWB_LP_ODR_DIV_800:
//    LpfCutFreq = AcceleroParams.AccOdr/800;
//    break;
//  default:
//    LpfCutFreq = 5000.0;
//    break;
//  }
//  MagSizeForLpf = (uint16_t)(LpfCutFreq / AccMagResults.BinFreqStep);
//  
//  /* Set the mag size to be used */
//  AccMagResults.MagSizeTBU = MagSizeForLpf;
//#endif /* LIMIT_FFT_RESULT */
  
  return BSP_Error;
}

/**
  * @brief  Enable/Disable the accelerometer HP Filter
  * @param  Cutoff frequency
  * @retval 0 in case of success
  */
uint8_t EnableDisable_ACC_HP_Filter(uint8_t FilterIsEnabled)
{
  int32_t BSP_Error = BSP_ERROR_NONE;
  
  if(FilterIsEnabled)
  {
    if ((BSP_Error = MOTION_SENSOR_Enable_HP_Filter(ACCELERO_INSTANCE, AcceleroParams.HwFilter)) != BSP_ERROR_NONE)
    {
      PREDMNT1_PRINTF("\r\nError Enable/Disable HP Filter (BSP_ERROR = %ld)\r\n", BSP_Error);
      return BSP_Error;
    }
    else
    {
      PREDMNT1_PRINTF("\r\nEnable HP Filter\r\n\t--> OK\r\n");
    }
  }
  else
  {
    if( (BSP_Error = MOTION_SENSOR_Write_Register(ACCELERO_INSTANCE, ISM330DHCX_CTRL8_XL, 0x00)) != BSP_ERROR_NONE)
    {
      PREDMNT1_PRINTF("\r\nError Disable HP Filter (BSP_ERROR = %ld)", BSP_Error);
      return BSP_Error;
    }
    else
    {
      PREDMNT1_PRINTF("\r\nDisable HP Filter\r\n\t--> OK\r\n");
    }
  }
  
  return BSP_Error;
}

/**
  * @brief  MotionSP TD Threshold Updating
  * @param  dflt Force updating with default values
  * @retval Error codes
  */
void MotionSP_TD_Threshold_Updating(uint8_t dflt)
{
  if (dflt)
  {
    memcpy((void *)TDRmsThresh, (void *)&TDSpeedRMSThresh, sizeof(sTimeDomainThresh_t));
    memcpy((void *)TDPkThresh, (void *)&TDAccPeakThresh, sizeof(sTimeDomainThresh_t));
  }
  else
  {
    __NOP();
  }
}

/**
  * @brief  MotionSP FD Threshold Updating
  * @param  subrange_num Subrange number to be used
  * @param  dflt Force updating with default values
  * @retval Error codes
  */
void MotionSP_FD_Threshold_Updating(uint8_t subrange_num, uint8_t dflt)
{
  float *pFDWarnThresh_Sub = NULL;
  float *pFDAlarmThresh_Sub= NULL;
  
  uint16_t thr_array_size = 0;
  
  thr_array_size = 3 * subrange_num * sizeof(float);
  
  if (dflt)
  {
    switch (subrange_num)
    {
    case 8:
      pFDWarnThresh_Sub = (float *)FDWarnThresh_Sub8;
      pFDAlarmThresh_Sub = (float *)FDAlarmThresh_Sub8;
      break;
      
    case 16:
      pFDWarnThresh_Sub = (float *)FDWarnThresh_Sub16;
      pFDAlarmThresh_Sub = (float *)FDAlarmThresh_Sub16;
      break;
      
    case 32:
      pFDWarnThresh_Sub = (float *)FDWarnThresh_Sub32;
      pFDAlarmThresh_Sub = (float *)FDAlarmThresh_Sub32;
      break;
      
    case 64:
      pFDWarnThresh_Sub = (float *)FDWarnThresh_Sub64;
      pFDAlarmThresh_Sub = (float *)FDAlarmThresh_Sub64;
      break;
    }
    
    memcpy((void *)FDWarnThresh_Sub, (void *)pFDWarnThresh_Sub, thr_array_size);
    memcpy((void *)FDAlarmThresh_Sub, (void *)pFDAlarmThresh_Sub, thr_array_size);
  }
  else
  {
    __NOP();
  }
}

/**
  * @brief 	Routine to be executed on IRQ about accelerometer data ready
  * @return None
  */    
void MotionSP_DataReady_IRQ_Rtn(void)
{
  AccDrdyNr++;
}

/**
  * @brief 	Routine to be executed on IRQ about accelerometer fifo full
  * @return None
  */    
void MotionSP_FifoFull_IRQ_Rtn(void)
{
  // Read all FIFO data
  MOTION_SENSOR_FIFO_Read(ACCELERO_INSTANCE, MotionSP_Data.FifoBfr, AcceleroParams.FifoWtm);
  
  // Create circular buffer from accelerometer FIFO
  FillAccCircBuffFromFifo(&MotionSP_Data.AccCircBuff, MotionSP_Data.FifoBfr, AcceleroParams.FifoWtm, AcceleroParams.AccSens, RestartFlag);

  // Fifo has been read
  MotionSP_Data.FifoEmpty = 1;
}

/*------------------ Predictive APIs -------------------------*/

/**
  *  @brief Initialization of WARNING & ALARM thresholds values on Axes,
  *         using the default values included in MotionSP_Thresholds.h file
  *  @param pTdAlarm: Pointer to TimeDomain Alarms Result
  *  @param pTimeDomainVal: Pointer to TimeDomain Value Result
  *  @param pTdRmsThreshold: Pointer to TimeDomain RMS Threshlods to initialize
  *  @param pTdPkThreshold:  Pointer to TimeDomain PK Threshlods to initialize
  *  @return Return description
  */
void MotionSP_TimeDomainAlarmInit (sTimeDomainAlarm_t *pTdAlarm,
                                   sTimeDomainData_t *pTimeDomainVal,
                                   sTimeDomainThresh_t *pTdRmsThreshold,
                                   sTimeDomainThresh_t *pTdPkThreshold) 
{
  /* Reset status value for TimeDomain Alarms Result */
  memset(pTdAlarm, NONE, sizeof(sTimeDomainAlarm_t));
  
  /* Reset status value for TimeDomain Value Result */
  memset(pTimeDomainVal, 0, sizeof(sTimeDomainData_t));
  
  memcpy((void *)pTdRmsThreshold, (void *)TDRmsThresh, sizeof(sTimeDomainThresh_t));
  memcpy((void *)pTdPkThreshold, (void *)TDPkThresh, sizeof(sTimeDomainThresh_t));
}

/**
  *  @brief  Time Domain Alarm Analysis based just on Speed RMS FAST Moving Average
  *  @param  pTdAlarm: Pointer to TimeDomain Alarms Result
  *  @param  pTimeDomainVal: Pointer to TimeDomain Value Result
  *  @param  pTdRmsThreshold:  Pointer to TimeDomain RMS Threshlods Configured
  *  @param  pTdPkThreshold:  Pointer to TimeDomain PK Threshlods Configured
  *  @param  pTimeDomain:   Pointer to TimeDomain Parameters to monitor
  *  @return None
  */
void MotionSP_TimeDomainAlarm (sTimeDomainAlarm_t *pTdAlarm,
                               sTimeDomainData_t *pTimeDomainVal,
                               sTimeDomainThresh_t *pTdRmsThreshold,
                               sTimeDomainThresh_t *pTdPkThreshold,
                               sTimeDomainData_t *pTimeDomain) 
{
  /* Reset status value for Time Domain alarms */
  memset(pTdAlarm, GOOD, 3*2);
  
  pTimeDomainVal->SpeedRms.AXIS_X = pTimeDomain->SpeedRms.AXIS_X*1000;
  pTimeDomainVal->SpeedRms.AXIS_Y = pTimeDomain->SpeedRms.AXIS_Y*1000;
  pTimeDomainVal->SpeedRms.AXIS_Z = pTimeDomain->SpeedRms.AXIS_Z*1000;
  
  /* Speed RMS comparison with thresholds */      
  if ((pTimeDomain->SpeedRms.AXIS_X*1000) > pTdRmsThreshold->THR_WARN_AXIS_X)
  {
    pTdAlarm->RMS_STATUS_AXIS_X = WARNING;
    pTimeDomainVal->SpeedRms.AXIS_X = pTimeDomain->SpeedRms.AXIS_X*1000;
  }
  if ((pTimeDomain->SpeedRms.AXIS_Y*1000) > pTdRmsThreshold->THR_WARN_AXIS_Y)
  {
    pTdAlarm->RMS_STATUS_AXIS_Y = WARNING;
    pTimeDomainVal->SpeedRms.AXIS_Y = pTimeDomain->SpeedRms.AXIS_Y*1000;
  }
  if ((pTimeDomain->SpeedRms.AXIS_Z*1000) > pTdRmsThreshold->THR_WARN_AXIS_Z)
  {
    pTdAlarm->RMS_STATUS_AXIS_Z = WARNING;
    pTimeDomainVal->SpeedRms.AXIS_Z = pTimeDomain->SpeedRms.AXIS_Z*1000;
  }
  if ((pTimeDomain->SpeedRms.AXIS_X*1000) > pTdRmsThreshold->THR_ALARM_AXIS_X)
  {
    pTdAlarm->RMS_STATUS_AXIS_X = ALARM;
    pTimeDomainVal->SpeedRms.AXIS_X = pTimeDomain->SpeedRms.AXIS_X*1000;
  }
  if ((pTimeDomain->SpeedRms.AXIS_Y*1000) > pTdRmsThreshold->THR_ALARM_AXIS_Y)
  {
    pTdAlarm->RMS_STATUS_AXIS_Y = ALARM;
    pTimeDomainVal->SpeedRms.AXIS_Y = pTimeDomain->SpeedRms.AXIS_Y*1000;
  }
  if ((pTimeDomain->SpeedRms.AXIS_Z*1000) > pTdRmsThreshold->THR_ALARM_AXIS_Z)
  {
    pTdAlarm->RMS_STATUS_AXIS_Z = ALARM;
    pTimeDomainVal->SpeedRms.AXIS_Z = pTimeDomain->SpeedRms.AXIS_Z*1000;
  }
  
  pTimeDomainVal->AccPeak.AXIS_X = pTimeDomain->AccPeak.AXIS_X;
  pTimeDomainVal->AccPeak.AXIS_Y = pTimeDomain->AccPeak.AXIS_Y;
  pTimeDomainVal->AccPeak.AXIS_Z = pTimeDomain->AccPeak.AXIS_Z;
  
  /* Accelerometer Peak comparison with thresholds */      
  if ((pTimeDomain->AccPeak.AXIS_X) > pTdPkThreshold->THR_WARN_AXIS_X)
  {
    pTdAlarm->PK_STATUS_AXIS_X = WARNING;
    pTimeDomainVal->AccPeak.AXIS_X = pTimeDomain->AccPeak.AXIS_X;
  }
  if ((pTimeDomain->AccPeak.AXIS_Y) > pTdPkThreshold->THR_WARN_AXIS_Y)
  {
    pTdAlarm->PK_STATUS_AXIS_Y = WARNING;
    pTimeDomainVal->AccPeak.AXIS_Y = pTimeDomain->AccPeak.AXIS_Y;
  }
  if ((pTimeDomain->AccPeak.AXIS_Z) > pTdPkThreshold->THR_WARN_AXIS_Z)
  {
    pTdAlarm->PK_STATUS_AXIS_Z = WARNING;
    pTimeDomainVal->AccPeak.AXIS_Z = pTimeDomain->AccPeak.AXIS_Z;
  }
  if ((pTimeDomain->AccPeak.AXIS_X) > pTdPkThreshold->THR_ALARM_AXIS_X)
  {
    pTdAlarm->PK_STATUS_AXIS_X = ALARM;
    pTimeDomainVal->AccPeak.AXIS_X = pTimeDomain->AccPeak.AXIS_X;
  }
  if ((pTimeDomain->AccPeak.AXIS_Y) > pTdPkThreshold->THR_ALARM_AXIS_Y)
  {
    pTdAlarm->PK_STATUS_AXIS_Y = ALARM;
    pTimeDomainVal->AccPeak.AXIS_Y = pTimeDomain->AccPeak.AXIS_Y;
  }
  if ((pTimeDomain->AccPeak.AXIS_Z) > pTdPkThreshold->THR_ALARM_AXIS_Z)
  {
    pTdAlarm->PK_STATUS_AXIS_Z = ALARM;
    pTimeDomainVal->AccPeak.AXIS_Z = pTimeDomain->AccPeak.AXIS_Z;
  }
}

/**
  * @brief Initialization of Alarm Status on Axes, Alarm Values Reported
  *        and Thresholds to detect WARNING and ALARM conditions
  * @param pWarnThresh: Pointer to TimeDomain Alarms thresholds to use
  * @param pAlarmThresh: Pointer to TimeDomain Alarms thresholds to use
  * @param pTHR_Fft_Alarms: Pointer to Freq Domain Value Arrays Result
  * @param subrange_num:  Subranges numbers
  * @retval BSP status
  *  
  */
int32_t MotionSP_FreqDomainAlarmInit (float **pWarnThresh,
                                   float **pAlarmThresh,
                                   sFreqDomainAlarm_t *pTHR_Fft_Alarms,
                                   uint8_t subrange_num) 
{
  uint16_t thr_array_size = 0;
  
  thr_array_size = 3 * subrange_num * sizeof(float);
  
  /* Reset status value for FFT alarms */
  memset(pTHR_Fft_Alarms, NONE, sizeof(sFreqDomainAlarm_t));
  
  // Memory allocation for Warning Threshold array 
  if (*pWarnThresh == NULL)
    *pWarnThresh = (float *)malloc(thr_array_size);
  else
   *pWarnThresh = realloc(*pWarnThresh, thr_array_size);
  if (*pWarnThresh == NULL)
    return BSP_ERROR_MALLOC_FAILURE;

  // Memory allocation for Alarm Threshold array
  if (*pAlarmThresh == NULL)
    *pAlarmThresh = (float *)malloc(thr_array_size);
  else
   *pAlarmThresh = realloc(*pAlarmThresh, thr_array_size);
  if (*pAlarmThresh == NULL)
    return BSP_ERROR_MALLOC_FAILURE;

  memcpy((void *)*pWarnThresh, (void *)FDWarnThresh_Sub, thr_array_size);
  memcpy((void *)*pAlarmThresh, (void *)FDAlarmThresh_Sub, thr_array_size);
  
  return BSP_ERROR_NONE;
}

/**
  *  @brief  Compare the Frequency domain subrange comparison with external Threshold Arrays
  *  @param  pSRAmplitude: Pointer to Amplitude subranges Array resulting after Freq Analysis
  *  @param  pFDWarnThresh: Pointer to Amplitude Warning Threshold subranges Array
  *  @param  pFDAlarmThresh: Pointer to Amplitude Alarm Threshold subranges Array
  *  @param  subrange_num: Subranges number
  *  @param  pTHR_Check: Pointer to Amplitude subranges Values that exceed thresholds
  *  @param  pTHR_Fft_Alarms: Pointer to Amplitude subranges Threshold Status
  *  @return None
  */
void MotionSP_FreqDomainAlarm (sSubrange_t *pSRAmplitude,
                               float *pFDWarnThresh,
                               float *pFDAlarmThresh,
                               uint8_t subrange_num, 
                               sSubrange_t *pTHR_Check, 
                               sFreqDomainAlarm_t *pTHR_Fft_Alarms)
{
  float warn_thresholds;
  float alarm_thresholds;
  memset((void *)pTHR_Check, 0x00, 3*SUBRANGE_MAX * sizeof(float) );

  for(int i=0; i<subrange_num; i++)
  {
   for(int j=0; j<3; j++) 
   {
    warn_thresholds = *(pFDWarnThresh+(i*3)+j);
    alarm_thresholds = *(pFDAlarmThresh+(i*3)+j);
    switch (j)
    {
    case 0x00:  /* Axis X */
      pTHR_Check->AXIS_X[i] = pSRAmplitude->AXIS_X[i];
      pTHR_Fft_Alarms->STATUS_AXIS_X[i] = GOOD;       
      if(pSRAmplitude->AXIS_X[i] > warn_thresholds)
        pTHR_Fft_Alarms->STATUS_AXIS_X[i] = WARNING;
      if(pSRAmplitude->AXIS_X[i] > alarm_thresholds)
        pTHR_Fft_Alarms->STATUS_AXIS_X[i] = ALARM;
      break;
      
    case 0x01:  /* Axis Y */
      pTHR_Check->AXIS_Y[i] = pSRAmplitude->AXIS_Y[i];
      pTHR_Fft_Alarms->STATUS_AXIS_Y[i] = GOOD;
      if(pSRAmplitude->AXIS_Y[i] > warn_thresholds)
        pTHR_Fft_Alarms->STATUS_AXIS_Y[i] = WARNING;
      if(pSRAmplitude->AXIS_Y[i] > alarm_thresholds)
        pTHR_Fft_Alarms->STATUS_AXIS_Y[i] = ALARM;
      break;
      
    case 0x02:  /* Axis Z */
      pTHR_Check->AXIS_Z[i] = pSRAmplitude->AXIS_Z[i];
      pTHR_Fft_Alarms->STATUS_AXIS_Z[i] = GOOD;
      if(pSRAmplitude->AXIS_Z[i] > warn_thresholds)
        pTHR_Fft_Alarms->STATUS_AXIS_Z[i] = WARNING;
      if(pSRAmplitude->AXIS_Z[i] > alarm_thresholds)
        pTHR_Fft_Alarms->STATUS_AXIS_Z[i] = ALARM;
      break;
      
    default:
      __NOP();
      break;    
    }
   } 
  }
}

/**
  *  @brief Compare the Frequency domain subrange comparison with external Threshold Arrays
  *  @param pTdAlarm: Pointer to status array threshold resulting after Time Domain Analysis
  *  @param pTHR_Fft_Alarms: Pointer to status array threshold resulting after Freq Domain Analysis
  *  @param subrange_num: Subranges number
  *  @param pTotalTDAlarm: Pointer to total time domain alarms that exceed thresholds
  *  @param pTotalFDAlarm: Pointer to total frequency domain alarms that exceed thresholds
  *  @return None
  */
void MotionSP_TotalStatusAlarm(sTimeDomainAlarm_t *pTdAlarm,
                               sFreqDomainAlarm_t *pTHR_Fft_Alarms,
                               uint8_t subrange_num,
                               Alarm_Type_t *pTotalTDAlarm,
                               Alarm_Type_t *pTotalFDAlarm)
{
 Alarm_Type_t TempAlarm = GOOD;
 Alarm_Type_t TempFDAlarm = GOOD;
 
 TempAlarm = MAX4(TempAlarm,
                  pTdAlarm->PK_STATUS_AXIS_X,
                  pTdAlarm->PK_STATUS_AXIS_Y,
                  pTdAlarm->PK_STATUS_AXIS_Z);
 
 
 TempAlarm = MAX4(TempAlarm,
                  pTdAlarm->RMS_STATUS_AXIS_X,
                  pTdAlarm->RMS_STATUS_AXIS_Y,
                  pTdAlarm->RMS_STATUS_AXIS_Z);
 
 for(int i=0; i<subrange_num; i++)
 {
   TempFDAlarm = MAX4(TempFDAlarm,
                      pTHR_Fft_Alarms->STATUS_AXIS_X[i],
                      pTHR_Fft_Alarms->STATUS_AXIS_Y[i],
                      pTHR_Fft_Alarms->STATUS_AXIS_Z[i]);
 }
 
 *pTotalTDAlarm = TempAlarm;
 *pTotalFDAlarm = TempFDAlarm; 
  
}


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

/************************ (C) COPYRIGHT 2021 STMicroelectronics *****END OF FILE****/
