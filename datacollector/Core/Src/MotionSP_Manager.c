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
#include "BLE_Manager.h"
#include "BLE_ManagerCommon.h"

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
static uint16_t accCircBuffIndexForFftDelta = 0;
static uint8_t accCircBuffIndexForFftOvf = 0;

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

/* For BLE connectivity */
static uint8_t  SendingFFT= 0;
static uint8_t  MemoryIsAlloc= 0;
static uint16_t CountSendData= 0;

uint8_t *TotalBuffToSending;

volatile uint32_t FFT_Amplitude= 0;
volatile uint32_t FFT_Alarm=    0;

/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/
static uint8_t MotionSP_VibrationAnalysisManager(void);
static int32_t MotionSP_VibrationAnalysisVariableInit(void);
static int32_t MotionSP_AccMeasInit(void);
static int32_t AccOdrMeas(sAcceleroODR_t *pAcceleroODR);
static int32_t MotionSP_AcceleroFifoConfig(void);
static uint8_t MotionSP_FrequencyDomainAnalysis(void);
static void FillAccCircBuffFromFifo(sCircBuff_t *pAccCircBuff, uint8_t *pFifoBuff, uint16_t FifoWtm, float AccSens, uint8_t Rst);

static void PrepareTotalBuffToSending(sAccMagResults_t *ArrayToSend, uint16_t ActualMagSize);
static void BLE_AllarmUpdate(void);

/** @addtogroup STM32_MOTIONSP_MANAGER_PRIVATE_FUNCTIONS MotionSP Manager Private Functions
  * @{
  */

/**
  * @brief  Vibration Analysis
  * @retval None
  */
static uint8_t MotionSP_VibrationAnalysisManager(void)
{
  uint8_t VibrationAnalysisComplited= 0;
  
  if(MotionSP_Data.FifoEmpty)
  {
    /* Reset the TimeDomainData Drdy */
    MotionSP_Data.FifoEmpty = 0;
    
    /* Time Domain Analysis */
    MotionSP_TimeDomainEvalFromCircBuff(&TimeDomainData,
                                        &MotionSP_Data.AccCircBuff,
                                        AcceleroParams.FifoWtm,
                                        (Td_Type_t)MotionSP_Parameters.td_type,
                                        AcceleroODR,
                                        RestartFlag);
    
    /* Status check during Time domain Analysis */
    MotionSP_TimeDomainAlarm(&sTdAlarm,
                             &sTimeDomainVal,
                             &sTdRmsThresholds,
                             &sTdPkThresholds,
                             &TimeDomainData);    

    /* Frequency Domain Analysis */
    VibrationAnalysisComplited= MotionSP_FrequencyDomainAnalysis();
    
    if (RestartFlag)
      RestartFlag = 0;
  }
  
  return VibrationAnalysisComplited;
}

/**
  * @brief  Init the MotionSP Vibration parameters
  * @retval BSP status
  */ 
static int32_t MotionSP_VibrationAnalysisVariableInit(void)
{
  PREDMNT1_PRINTF("\r\nMotionSP Vibration Variable Initialization");
  
  /* Check for right values */
  switch (MotionSP_Parameters.FftSize)
  {
  case FFT_SIZE_256:
  case FFT_SIZE_512:
  case FFT_SIZE_1024:
  case FFT_SIZE_2048:
    break;
  default:
    /* Send fault info to terminal */
    PREDMNT1_PRINTF("\r\n\t--> FFT size is out of range.\r\n");
    return BSP_ERROR_WRONG_PARAM;
  }
   
  if ( ((MotionSP_Parameters.FftOvl) >= FFT_OVL_MIN) && ((MotionSP_Parameters.FftOvl) <= FFT_OVL_MAX) )
  {
    // MotionSP TD Threshold Updating
    MotionSP_TD_Threshold_Updating(TD_Th_DFLT);
    TD_Th_DFLT = 0;
    
    // MotionSP FD Threshold Updating
    MotionSP_FD_Threshold_Updating(MotionSP_Parameters.subrange_num, FD_Th_DFLT);
    FD_Th_DFLT = 0;
    
    // MotionSP threshold init
    if(FFT_Alarm)
    {
      /* Initialization of Alarm Status on Axes, Alarm Values Reported
         and Thresholds to detect WARNING and ALARM conditions */
      MotionSP_AlarmThreshold_Init();
    }
    
    // Reset the FFT magnitude data
    AccMagResults.MagSize = 0;
    AccMagResults.MagSizeTBU = 0;
    AccMagResults.BinFreqStep = 0;
    memset((void *)&AccMagResults.Max, 0, 3 * sizeof(sMaxOnArray_t));
    memset((void *)&AccMagResults.FFT_Items, 0, 3 * sizeof(uint16_t));
    
    // Set magnitude size
    AccMagResults.MagSize = MotionSP_Parameters.FftSize/2;
    
    /* Calculate the FFT bin frequency increment */
    AccMagResults.BinFreqStep = (AcceleroODR.Frequency / 2) / AccMagResults.MagSize;
    
    /* Set the mag size to be used */
    AccMagResults.MagSizeTBU = AccMagResults.MagSize;
    
    // Memory allocation for FFT magnitude arrays
    if (AccMagResults.Array.X == NULL)
      AccMagResults.Array.X = (float *) malloc(sizeof(float) * AccMagResults.MagSize);
    else
      AccMagResults.Array.X = realloc(AccMagResults.Array.X, sizeof(float) * AccMagResults.MagSize);
    if (AccMagResults.Array.X == NULL)
      return BSP_ERROR_MALLOC_FAILURE;
    
    if (AccMagResults.Array.Y == NULL)
      AccMagResults.Array.Y = (float *) malloc(sizeof(float) * AccMagResults.MagSize);
    else
      AccMagResults.Array.Y = realloc(AccMagResults.Array.Y, sizeof(float) * AccMagResults.MagSize);
    if (AccMagResults.Array.Y == NULL)
      return BSP_ERROR_MALLOC_FAILURE;
    
    if (AccMagResults.Array.Z == NULL)
      AccMagResults.Array.Z = (float *) malloc(sizeof(float) * AccMagResults.MagSize);
    else
      AccMagResults.Array.Z = realloc(AccMagResults.Array.Z, sizeof(float) * AccMagResults.MagSize);
    if (AccMagResults.Array.Z == NULL)
      return BSP_ERROR_MALLOC_FAILURE;
    
    // Reset the TimeDomain parameter values
    memset((void *)&TimeDomainData, 0x00, sizeof(sTimeDomainData_t));
    
    // Create filter array related to windowing method
    MotionSP_SetWindFiltArray(Filter_Params, MotionSP_Parameters.FftSize, (Filt_Type_t)MotionSP_Parameters.window);

    // Initialize FFT seed
    arm_rfft_fast_init_f32(&fftS, MotionSP_Parameters.FftSize);

    /* It is the minimum value to do the first FFT */
    accCircBuffIndexForFft = MotionSP_Parameters.FftSize - 1;
    
    /* Reset variable used inside MotionSP_fftExecution() */
    accCircBuffIndexForFftDelta = (uint16_t)((float)(MotionSP_Parameters.FftSize)*((100.0f-(float)MotionSP_Parameters.FftOvl)/100.0f));
    accCircBuffIndexForFftOvf = 0;
    
    if(MemoryIsAlloc)
    {
      free(TotalBuffToSending);
      MemoryIsAlloc= 0;
    }
 
    RestartFlag = 1;
    SendingFFT= 0;
    
    PREDMNT1_PRINTF("\r\n\t--> OK\r\n");
  }
  else
  {
    /* Send fault info to terminal */
    PREDMNT1_PRINTF("\r\n\t--> Overlapping size is out of range.\r\n");
    
    return BSP_ERROR_WRONG_PARAM;
  }
  
  return BSP_ERROR_NONE;
}

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

///**
//  * @brief  Configure the FIFO in according with the algorithm setting
//  * @retval BSP status
//  */
//static int32_t MotionSP_AcceleroFifoConfig(void)
//{
//  int32_t BSP_Error = BSP_ERROR_NONE;
//  float k = 0.80f;
//  
//  // Reset the STM32 Motion Signal Processing Data
//  MotionSP_Data.FifoEmpty = 0;
//  MotionSP_Data.AccCircBuff.Size = 0;
//  MotionSP_Data.AccCircBuff.IdPos = 0;
//  MotionSP_Data.AccCircBuff.Ovf = 0;
//  
//  // Set the initial position of the accelero circular buffer
//  MotionSP_Data.AccCircBuff.IdPos = (uint16_t)(-1);
//  
///*------------- WATERMARK Setting chiedere a GR ---------*/
//  if (MotionSP_Parameters.FftSize == 1024)
//  {
//    if (MotionSP_Parameters.FftOvl < 25)
//      k = 0.50f;
//    else if (MotionSP_Parameters.FftOvl < 35)
//      k = 0.57f;
//    else if (MotionSP_Parameters.FftOvl < 45)
//      k = 0.65f;
//    else if (MotionSP_Parameters.FftOvl < 50)
//      k = 0.77f;
//  }
//  else if (MotionSP_Parameters.FftSize == 2048)
//  {
//    if (MotionSP_Parameters.FftOvl < 25)
//      k = 0.25f;
//    else if (MotionSP_Parameters.FftOvl < 35)
//      k = 0.28f;
//    else if (MotionSP_Parameters.FftOvl < 45)
//      k = 0.32f;
//    else if (MotionSP_Parameters.FftOvl < 55)
//      k = 0.38f;
//    else if (MotionSP_Parameters.FftOvl < 65)
//      k = 0.47f;
//    else if (MotionSP_Parameters.FftOvl < 70)
//      k = 0.61f;
//    else if (MotionSP_Parameters.FftOvl < 75)
//      k = 0.71f;
//  }
//  
//  AcceleroParams.FifoWtm = (uint16_t)(((float)MotionSP_Parameters.FftSize*(1.0f-((float)MotionSP_Parameters.FftOvl/100.0f)))*k);
//  
//  if ( (uint16_t)((AcceleroParams.FifoWtm*7) > (1024*3)) )
//    return BSP_ERROR_WRONG_PARAM;
//  
//  /* Set FIFO watermark */
//  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Watermark_Level(ACCELERO_INSTANCE, AcceleroParams.FifoWtm)) != BSP_ERROR_NONE)
//    return BSP_Error;
//
//  /* MCU memory allocation for FIFO buffer ************************************/
//  if (MotionSP_Data.FifoBfr == NULL)
//    MotionSP_Data.FifoBfr = (uint8_t *) malloc(sizeof(uint8_t) * (AcceleroParams.FifoWtm * 7));
//  else
//    MotionSP_Data.FifoBfr = realloc(MotionSP_Data.FifoBfr, sizeof(uint8_t) * (AcceleroParams.FifoWtm * 7));
//  if (MotionSP_Data.FifoBfr == NULL)
//    return BSP_ERROR_MALLOC_FAILURE;
//  
//  /* Set to zero the allocated memory for FIFO buffer */
//  memset((void *)MotionSP_Data.FifoBfr, 0, sizeof(uint8_t) * (AcceleroParams.FifoWtm * 7));
//  /****************************************************************************/
//
//  /**** Create circular buffer ************************************************/
//  MotionSP_Data.AccCircBuff.Size = (uint16_t)((MotionSP_Parameters.FftSize * CIRC_BUFFER_RATIO_NUM)/CIRC_BUFFER_RATIO_DEN);
//  
//  /* MCU memory allocation for X-axis array for circular buffer ***************/
//  if (MotionSP_Data.AccCircBuff.Array.X == NULL)
//    MotionSP_Data.AccCircBuff.Array.X = (float *) malloc(sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  else
//   MotionSP_Data.AccCircBuff.Array.X = realloc(MotionSP_Data.AccCircBuff.Array.X, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  if(MotionSP_Data.AccCircBuff.Array.X == NULL)
//    return BSP_ERROR_MALLOC_FAILURE;
//
//  /* Set to zero the allocated memory for X-axis array for circular buffer */
//  memset((void *)MotionSP_Data.AccCircBuff.Array.X, 0, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  /****************************************************************************/
//  
//  /* MCU memory allocation for Y-axis array for circular buffer ***************/
//  if (MotionSP_Data.AccCircBuff.Array.Y == NULL)
//    MotionSP_Data.AccCircBuff.Array.Y = (float *) malloc(sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  else
//   MotionSP_Data.AccCircBuff.Array.Y = realloc(MotionSP_Data.AccCircBuff.Array.Y, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  if(MotionSP_Data.AccCircBuff.Array.Y == NULL)
//    return BSP_ERROR_MALLOC_FAILURE;
//
//  /* Set to zero the allocated memory for Y-axis array for circular buffer */
//  memset((void *)MotionSP_Data.AccCircBuff.Array.Y, 0, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  /****************************************************************************/
//  
//  /* MCU memory allocation for Z-axis array for circular buffer ***************/
//  if (MotionSP_Data.AccCircBuff.Array.Z == NULL)
//    MotionSP_Data.AccCircBuff.Array.Z = (float *) malloc(sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  else
//   MotionSP_Data.AccCircBuff.Array.Z = realloc(MotionSP_Data.AccCircBuff.Array.Z, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  if(MotionSP_Data.AccCircBuff.Array.Z == NULL)
//    return BSP_ERROR_MALLOC_FAILURE;
//
//  /* Set to zero the allocated memory for Z-axis array for circular buffer */
//  memset((void *)MotionSP_Data.AccCircBuff.Array.Z, 0, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
//  /****************************************************************************/
//  /****************************************************************************/
//  
//  /* Set FIFO depth to be limited to watermark threshold level  */
//  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Stop_On_Fth(ACCELERO_INSTANCE, ENABLE)) != BSP_ERROR_NONE)
//    return BSP_Error;
//
//  /* Set FIFO mode */
//  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Mode(ACCELERO_INSTANCE, ACCELERO_FIFO_STREAM_MODE)) != BSP_ERROR_NONE)
//    return BSP_Error;
//
//  /* IIS3DWB INT2_FIFO_FULL interrupt enable */
//  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_INT2_FIFO_Full(ACCELERO_INSTANCE, ENABLE)) != BSP_ERROR_NONE)
//    return BSP_Error;
//
//  /* Start acquisition time */
//  StartTick = BSP_GetTick();
//  
//  /* Set FIFO BDR value */
//  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_ODR_Value(ACCELERO_INSTANCE, AcceleroParams.AccFifoBdr)) != BSP_ERROR_NONE)
//    return BSP_Error;
//  
//  return BSP_Error;
//}

/**
  * @brief  Configure the FIFO in according with the algorithm setting
  * @retval BSP status
  */
static int32_t MotionSP_AcceleroFifoConfig(void)
{
  int32_t BSP_Error = BSP_ERROR_NONE;
  float k = 0.80f;
  
  PREDMNT1_PRINTF("\r\nAccelero FIFO config:\r\n");
  
  // Reset the STM32 Motion Signal Processing Data
  MotionSP_Data.FifoEmpty = 0;
  MotionSP_Data.AccCircBuff.Size = 0;
  MotionSP_Data.AccCircBuff.IdPos = 0;
  MotionSP_Data.AccCircBuff.Ovf = 0;
  
  // Set the initial position of the accelero circular buffer
  MotionSP_Data.AccCircBuff.IdPos = (uint16_t)(-1);
  
/*------------- WATERMARK Setting ---------*/
  if (MotionSP_Parameters.FftSize == 1024)
  {
    if (MotionSP_Parameters.FftOvl < 20)
      k = 0.66f;   
  }
  else
  {
    if (MotionSP_Parameters.FftOvl < 20)
      k = 0.33f;
    else if (MotionSP_Parameters.FftOvl < 40)
      k = 0.40f;
    else if (MotionSP_Parameters.FftOvl < 60)
      k = 0.50f;
  }
  
  AcceleroParams.FifoWtm = (uint16_t)(((float)MotionSP_Parameters.FftSize*(1.0f-((float)MotionSP_Parameters.FftOvl/100.0f)))*k);
    
  if ( (uint16_t)((AcceleroParams.FifoWtm * FIFO_FRAME_SIZE) > MAX_FIFO_SIZE) )
  { PREDMNT1_PRINTF("\n FIFO size will be %i byte than exceed its maximum value, 3 kbyte.\n",
                    (uint16_t)(AcceleroParams.FifoWtm * FIFO_FRAME_SIZE));
    PREDMNT1_PRINTF(" Please, reduce the FFT_SIZE_DEFAULT or the FFT_OVL_DEFAULT. \n");
    return BSP_ERROR_WRONG_PARAM;
  }
  /* Set FIFO watermark */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Watermark_Level(ACCELERO_INSTANCE, AcceleroParams.FifoWtm)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError Set FIFO Watermark Level (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk Set FIFO Watermark Level\r\n");
  }

  /** MCU memory allocation for FIFO buffer ***********************************/
  if (MotionSP_Data.FifoBfr == NULL)
    MotionSP_Data.FifoBfr = (uint8_t *) malloc(sizeof(uint8_t) * (AcceleroParams.FifoWtm * FIFO_FRAME_SIZE));
  else
    MotionSP_Data.FifoBfr = realloc(MotionSP_Data.FifoBfr, sizeof(uint8_t) * (AcceleroParams.FifoWtm * FIFO_FRAME_SIZE));
  if (MotionSP_Data.FifoBfr == NULL)
    return BSP_ERROR_MALLOC_FAILURE;
  
  /* Set to zero the allocated memory for FIFO buffer */
  memset((void *)MotionSP_Data.FifoBfr, 0, sizeof(uint8_t) * (AcceleroParams.FifoWtm * FIFO_FRAME_SIZE));
  /****************************************************************************/

  /**** Create circular buffer ************************************************/
  MotionSP_Data.AccCircBuff.Size = (uint16_t)((MotionSP_Parameters.FftSize * CIRC_BUFFER_RATIO_NUM)/CIRC_BUFFER_RATIO_DEN);
  
  /* MCU memory allocation for X-axis array for circular buffer ***************/
  if (MotionSP_Data.AccCircBuff.Array.X == NULL)
    MotionSP_Data.AccCircBuff.Array.X = (float *) malloc(sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  else
   MotionSP_Data.AccCircBuff.Array.X = realloc(MotionSP_Data.AccCircBuff.Array.X, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  if(MotionSP_Data.AccCircBuff.Array.X == NULL)
    return BSP_ERROR_MALLOC_FAILURE;

  /* Set to zero the allocated memory for X-axis array for circular buffer */
  memset((void *)MotionSP_Data.AccCircBuff.Array.X, 0, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  /****************************************************************************/
  
  /** MCU memory allocation for Y-axis array for circular buffer **************/
  if (MotionSP_Data.AccCircBuff.Array.Y == NULL)
    MotionSP_Data.AccCircBuff.Array.Y = (float *) malloc(sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  else
   MotionSP_Data.AccCircBuff.Array.Y = realloc(MotionSP_Data.AccCircBuff.Array.Y, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  if(MotionSP_Data.AccCircBuff.Array.Y == NULL)
    return BSP_ERROR_MALLOC_FAILURE;

  /* Set to zero the allocated memory for Y-axis array for circular buffer */
  memset((void *)MotionSP_Data.AccCircBuff.Array.Y, 0, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  /****************************************************************************/
  
  /** MCU memory allocation for Z-axis array for circular buffer **************/
  if (MotionSP_Data.AccCircBuff.Array.Z == NULL)
    MotionSP_Data.AccCircBuff.Array.Z = (float *) malloc(sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  else
   MotionSP_Data.AccCircBuff.Array.Z = realloc(MotionSP_Data.AccCircBuff.Array.Z, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  if(MotionSP_Data.AccCircBuff.Array.Z == NULL)
    return BSP_ERROR_MALLOC_FAILURE;

  /* Set to zero the allocated memory for Z-axis array for circular buffer */
  memset((void *)MotionSP_Data.AccCircBuff.Array.Z, 0, sizeof(float) * MotionSP_Data.AccCircBuff.Size);
  /****************************************************************************/
  
  /* Set FIFO depth to be limited to watermark threshold level  */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Stop_On_Fth(ACCELERO_INSTANCE, ENABLE)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError Set FIFO Stop On Fth (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk Set FIFO Stop On Fth\r\n");
  }
  
    /* Enable AXL data to FIFO with no decimation */
//  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Decimation(ACCELERO_INSTANCE,
//                                                         MOTION_ACCELERO,
//                                                        (uint8_t)ISM330DLC_FIFO_XL_NO_DEC)) != BSP_ERROR_NONE)
//  {
//    PREDMNT1_PRINTF("\tError Set FIFO Decimation (BSP_ERROR = %ld)\r\n", BSP_Error);
//    return BSP_Error;
//  }
//  else
//    PREDMNT1_PRINTF("\tOk Set FIFO Decimation\r\n");
   
  return BSP_Error;
}

/**
  * @brief  Frequency Domain Analysis
  * @retval None
  */
static uint8_t MotionSP_FrequencyDomainAnalysis(void)
{
  uint16_t accCircBuffIndexTmp = 0;
  uint16_t accCircBuffIndexForFftTmp = 0;
  uint8_t FFT_avg = 0;
  
  accCircBuffIndexTmp = MotionSP_Data.AccCircBuff.IdPos + (MotionSP_Data.AccCircBuff.Ovf * MotionSP_Data.AccCircBuff.Size);
  accCircBuffIndexForFftTmp = accCircBuffIndexForFft + (accCircBuffIndexForFftOvf * MotionSP_Data.AccCircBuff.Size);
  
  if (accCircBuffIndexTmp >= accCircBuffIndexForFftTmp)
  {
    if ((BSP_GetTick() - StartTick) > MotionSP_Parameters.tacq)
    {
      FFT_avg = 1;
      disable_FIFO();
    }
    
    MotionSP_fftExecution(FFT_avg);
    
    accCircBuffIndexForFftOvf = 0;
    accCircBuffIndexForFft += accCircBuffIndexForFftDelta;
    if (accCircBuffIndexForFft >= MotionSP_Data.AccCircBuff.Size)
    {
      accCircBuffIndexForFft -= MotionSP_Data.AccCircBuff.Size;
      
      if (!MotionSP_Data.AccCircBuff.Ovf)
        accCircBuffIndexForFftOvf = 1;
    }
    
    MotionSP_Data.AccCircBuff.Ovf = 0;
  }
  
  return FFT_avg;
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
  * @brief  Prepare buffer to send data to Bluetooth
  * @param  sAccMagResults_t *ArrayToSend X, Y and Z FFT Amplitude values
  * @param  uint16_t ActualMagSize Number of samples for each component
  * @retval None
  */
static void PrepareTotalBuffToSending(sAccMagResults_t *ArrayToSend, uint16_t ActualMagSize)
{
  uint32_t BuffPos;
  uint16_t TotalSize;
  
  float TempFloat;
  uint32_t *TempBuff = (uint32_t *) & TempFloat;
  
  TotalSize= 2 /* nSample */ + 1 /* nComponents */ + 4 /*  Frequency Steps */ + ((3 * ActualMagSize) * 4) /* Samples */;
  TotalBuffToSending=(uint8_t *)malloc(TotalSize);
	
  if(TotalBuffToSending == NULL)
  {
    Error_Handler();
  }
	
  MemoryIsAlloc= 1;
    
  STORE_LE_16(TotalBuffToSending  ,ActualMagSize);
  TotalBuffToSending[2]= 3;
  BuffPos= 3;
  
  TempFloat= AccMagResults.BinFreqStep;
  STORE_LE_32(&TotalBuffToSending[BuffPos], *TempBuff);
  BuffPos += 4;
  
  for(int i=0; i<ActualMagSize; i++)
  {
    STORE_LE_32(&TotalBuffToSending[BuffPos], *(uint32_t *)&(AccMagResults.Array.X[i]));
    BuffPos += 4;
  }
  
  for(int i=0; i<ActualMagSize; i++)
  {
    STORE_LE_32(&TotalBuffToSending[BuffPos], *(uint32_t *)&(AccMagResults.Array.Y[i]));
    BuffPos += 4;
  }
  
  for(int i=0; i<ActualMagSize; i++)
  {
    STORE_LE_32(&TotalBuffToSending[BuffPos], *(uint32_t *)&(AccMagResults.Array.Z[i]));
    BuffPos += 4;
  }
}

static void BLE_AllarmUpdate(void)
{  
  /***********************************
   * Send Speed RMS value and status *
   ***********************************/
  
  sBLE_Manager_FFTAlarmSpeedStatusAlarm_t AlarmSpeedStatus;
  BLE_MANAGER_FFTAlarmSpeedStatusGenericValue_t SpeedRmsValue;
  
  AlarmSpeedStatus.STATUS_AXIS_X= (BLE_Manager_FFTAlarmSpeedStatusAlarmType_t)sTdAlarm.RMS_STATUS_AXIS_X;
  AlarmSpeedStatus.STATUS_AXIS_Y= (BLE_Manager_FFTAlarmSpeedStatusAlarmType_t)sTdAlarm.RMS_STATUS_AXIS_Y;
  AlarmSpeedStatus.STATUS_AXIS_Z= (BLE_Manager_FFTAlarmSpeedStatusAlarmType_t)sTdAlarm.RMS_STATUS_AXIS_Z;
  
  SpeedRmsValue.x= sTimeDomainVal.SpeedRms.AXIS_X;
  SpeedRmsValue.y= sTimeDomainVal.SpeedRms.AXIS_Y;
  SpeedRmsValue.z= sTimeDomainVal.SpeedRms.AXIS_Z;
  
  PREDMNT1_PRINTF("Sending Speed RMS value and status to ST BLE Sensor app\r\n");
  BLE_FFTAlarmSpeedStatusUpdate(AlarmSpeedStatus, SpeedRmsValue);
    
  /**********************************
   * Send Acc Peak value and status *
   **********************************/
  
  sBLE_Manager_FFTAlarmAccPeakStatusAlarm_t AlarmAccPeakStatus;
  BLE_MANAGER_FFTAlarmAccPeakStatusGenericValue_t AccPeak;
  
  AlarmAccPeakStatus.STATUS_AXIS_X= (BLE_Manager_FFTAlarmAccPeakStatusAlarmType_t)sTdAlarm.PK_STATUS_AXIS_X;
  AlarmAccPeakStatus.STATUS_AXIS_Y= (BLE_Manager_FFTAlarmAccPeakStatusAlarmType_t)sTdAlarm.PK_STATUS_AXIS_Y;
  AlarmAccPeakStatus.STATUS_AXIS_Z= (BLE_Manager_FFTAlarmAccPeakStatusAlarmType_t)sTdAlarm.PK_STATUS_AXIS_Z;
  
  AccPeak.x= sTimeDomainVal.AccPeak.AXIS_X;
  AccPeak.y= sTimeDomainVal.AccPeak.AXIS_Y;
  AccPeak.z= sTimeDomainVal.AccPeak.AXIS_Z;
  
  PREDMNT1_PRINTF("Sending Acc Peak value and status to ST BLE Sensor app\r\n");
  BLE_FFTAlarmAccPeakStatusUpdate(AlarmAccPeakStatus, AccPeak);
    
  /*********************************************************************
   * Send the frequency domain threshold status for max Subrange value *
   *********************************************************************/
  
  sBLE_Manager_FFTAlarmSubrangeStatusAlarm_t AlarmSugrangeStatus;
  BLE_MANAGER_FFTAlarmSubrangeStatusGenericValue_t SendMaxValue;
  BLE_MANAGER_FFTAlarmSubrangeStatusGenericValue_t SendFreqMaxValue;
  float BinFreqStep;
  
  Alarm_Type_t TempAlarm_X= GOOD;
  Alarm_Type_t TempAlarm_Y= GOOD;
  Alarm_Type_t TempAlarm_Z= GOOD;
  
  for(int i=0; i<MotionSP_Parameters.subrange_num; i++)
  {
    TempAlarm_X= ALARM_MAX(TempAlarm_X, THR_Fft_Alarms.STATUS_AXIS_X[i]);
    TempAlarm_Y= ALARM_MAX(TempAlarm_Y, THR_Fft_Alarms.STATUS_AXIS_Y[i]);
    TempAlarm_Z= ALARM_MAX(TempAlarm_Z, THR_Fft_Alarms.STATUS_AXIS_Z[i]);
  }
  
  AlarmSugrangeStatus.STATUS_AXIS_X = (BLE_Manager_FFTAlarmSubrangeStatusAlarmType_t)TempAlarm_X;
  AlarmSugrangeStatus.STATUS_AXIS_Y = (BLE_Manager_FFTAlarmSubrangeStatusAlarmType_t)TempAlarm_Y;
  AlarmSugrangeStatus.STATUS_AXIS_Z = (BLE_Manager_FFTAlarmSubrangeStatusAlarmType_t)TempAlarm_Z;
  
  SendMaxValue.x= AccMagResults.Max.X.value;
  SendMaxValue.y= AccMagResults.Max.Y.value;
  SendMaxValue.z= AccMagResults.Max.Z.value;
  
  BinFreqStep = (AcceleroODR.Frequency /2 ) / AccMagResults.MagSize;
  SendFreqMaxValue.x= AccMagResults.Max.X.loc * BinFreqStep;
  SendFreqMaxValue.y= AccMagResults.Max.Y.loc * BinFreqStep;
  SendFreqMaxValue.z= AccMagResults.Max.Z.loc * BinFreqStep;
  
  PREDMNT1_PRINTF("Sending the frequency domain threshold status for max Subrange value to ST BLE Sensor app\r\n");
  BLE_FFTAlarmSubrangeStatusUpdate(AlarmSugrangeStatus, SendMaxValue, SendFreqMaxValue);
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
  
  uint32_t IntPart, DecPart;
  MCR_BLUEMS_F2I_2D(AcceleroODR.Frequency, IntPart, DecPart);

  /* Send the parameters to terminal */
  PREDMNT1_PRINTF(" %ld.%.2ld Hz )\r\n", IntPart, DecPart);
  
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
  * @brief  MotionSP Alarm Threshold Init
  * @retval Error codes
  */
void MotionSP_AlarmThreshold_Init(void)
{
  MotionSP_TimeDomainAlarmInit(&sTdAlarm,
                               &sTimeDomainVal,
                               &sTdRmsThresholds,
                               &sTdPkThresholds);
  
  if (MotionSP_FreqDomainAlarmInit(&pFDWarnThresh,
                                   &pFDAlarmThresh,
                                   &THR_Fft_Alarms,
                                   MotionSP_Parameters.subrange_num) != BSP_ERROR_NONE)
    Error_Handler();
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
  * @brief  Vibration analysis manager
  * @retval BSP status
  */
int32_t MotionSP_MainManager(void)
{
  int32_t  BSP_Error = BSP_ERROR_NONE;
  
  if (!MotionSP_Running)
  {
    /* Condition Monitoring Initialization */
    
    /* Initializes the vibration parameters with default values */
    if ((BSP_Error = MotionSP_VibrationAnalysisVariableInit()) != BSP_ERROR_NONE)
      return BSP_Error;
  
    /* Enable the HP Filter */
    if((EnableDisable_ACC_HP_Filter(AcceleroParams.HwFilter)) != BSP_ERROR_NONE)
      return BSP_Error;

    /* Vibration Analysis Start */
    
    /* Configure the FIFO settings in according with others parammeters changed */
    if ((BSP_Error = MotionSP_AcceleroFifoConfig()) != BSP_ERROR_NONE)
      return BSP_Error;
    
    enable_FIFO();
    
    MotionSP_Running = 1;
  }
  else
  {
    static uint8_t Reset= 0;
    
    if(!SendingFFT)
    {
      uint8_t VibrationAnalysisComplited= 0;
      
      /* Frequency Domain Analysis */
      VibrationAnalysisComplited= MotionSP_VibrationAnalysisManager();
      
      /* Send data to ST BLE Sensor app*/
      if( (FFT_Amplitude) &&
          (VibrationAnalysisComplited) )
      {
        PrepareTotalBuffToSending(&AccMagResults, AccMagResults.MagSize);
        
        /* Send Time Domain to ST BLE Sensor app */
        PREDMNT1_PRINTF("\r\nSending Time Domain to ST BLE Sensor app\r\n");
        BLE_MANAGER_TimeDomainGenericValue_t AccPeak;
        BLE_MANAGER_TimeDomainGenericValue_t SpeedRms;
        
        AccPeak.x= TimeDomainData.AccPeak.AXIS_X;
        AccPeak.y= TimeDomainData.AccPeak.AXIS_Y;
        AccPeak.z= TimeDomainData.AccPeak.AXIS_Z;
        
        SpeedRms.x= TimeDomainData.SpeedRms.AXIS_X;
        SpeedRms.y= TimeDomainData.SpeedRms.AXIS_Y;
        SpeedRms.z= TimeDomainData.SpeedRms.AXIS_Z;
        
        BLE_TimeDomainUpdate(AccPeak, SpeedRms);
        
        /* Send Accelerometer ARRAYs FFT average values to ST BLE Sensor app */
        PREDMNT1_PRINTF("Sending FFT Amplitude to ST BLE Sensor app\r\n");
        SendingFFT= 1;
        CountSendData= 0;
      }
      
      if( (FFT_Alarm) &&
          (VibrationAnalysisComplited) )
      {
        /* Check thresholds for Accelerometer FFT Result values in SubRanges */
        MotionSP_FreqDomainAlarm (&SRAmplitude,
                                  pFDWarnThresh,
                                  pFDAlarmThresh,
                                  MotionSP_Parameters.subrange_num,
                                  &THR_Check, 
                                  &THR_Fft_Alarms);
        
        BLE_AllarmUpdate();
   
        //FFT_AlarmSubrangeStatus_Update(&AccMagResults,&THR_Fft_Alarms,MotionSP_Parameters.subrange_num,AccMagResults.MagSize);
        
        Reset= 1;
      }
    }
    else
    {
      /* Send Accelerometer ARRAYs FFT average values to ST BLE Sensor app */
      BLE_FFTAmplitudeUpdate(TotalBuffToSending, AccMagResults.MagSize, &SendingFFT, &CountSendData);
      
      if(!SendingFFT)
        Reset= 1;
    }
    
    if(Reset)
    {
      /* Re-Initializes the vibration parameters */
      if ((BSP_Error = MotionSP_VibrationAnalysisVariableInit()) != BSP_ERROR_NONE)
        return BSP_Error;
      
      enable_FIFO();
      
      Reset= 0;
    }
  }

  return BSP_Error;
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
  * @brief  Enable FIFO measuring
  * @param  None
  * @retval BSP status
  */
int32_t enable_FIFO(void)
{
  int32_t BSP_Error= BSP_ERROR_NONE;
  
  PREDMNT1_PRINTF("\r\nEnable FIFO\r\n");
  
  /* Set FIFO to STREAM Mode (Continuous) */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Mode(ACCELERO_INSTANCE, ACCELERO_FIFO_STREAM_MODE)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError Set FIFO in Continuous Mode (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk Set FIFO in Continuous Mode\r\n");
  }

  /* ISM330DLC INT2_FIFO_FULL interrupt enable */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_INT2_FIFO_Full(ACCELERO_INSTANCE, ENABLE)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError enable FIFO FULL INTP on INT2 path (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk enable FIFO FULL INTP on INT2 path\r\n");
  }
   
//  /* Set FIFO ODR value */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_BDR(ACCELERO_INSTANCE,
                                              MOTION_ACCELERO,
                                              AcceleroParams.AccFifoBdr)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError Set FIFO ODR Value (BSP_ERROR = %ld)\r\n", BSP_Error);
  }
  else
  {
    PREDMNT1_PRINTF("\tOk Set FIFO ODR Value\r\n");
  } 
  
  /* Start acquisition time */
  StartTick = BSP_GetTick();

  FifoEnabled = 1;
  
  return BSP_Error;
}

/**
  * @brief  Disable FIFO measuring
  * @param  None
  * @retval BSP status
  */
int32_t disable_FIFO(void)
{
  int32_t BSP_Error= BSP_ERROR_NONE;
  
  PREDMNT1_PRINTF("\r\nDisable FIFO\r\n");
  
  /* Switch from FIFO to BYPASS MODE */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_Mode(ACCELERO_INSTANCE, ACCELERO_FIFO_BYPASS_MODE)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError set FIFO in By-Pass Mode (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk set FIFO in By-Pass Mode\r\n");
  }

  /* ISM330DLC INT2_FIFO_FULL interrupt enable */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_INT2_FIFO_Full(ACCELERO_INSTANCE, DISABLE)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError disable FIFO FULL INTP on INT2 path (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk disable FIFO FULL INTP on INT2 path\r\n");
  }
  
  /* Switch off the FIFO filling */
  if ((BSP_Error = MOTION_SENSOR_FIFO_Set_BDR(ACCELERO_INSTANCE, MOTION_ACCELERO, 0.0)) != BSP_ERROR_NONE)
  {
    PREDMNT1_PRINTF("\tError Set FIFO ODR Value (BSP_ERROR = %ld)\r\n", BSP_Error);
    return BSP_Error;
  }
  else
  {
    PREDMNT1_PRINTF("\tOk Set FIFO ODR Value\r\n");
  } 

  FifoEnabled = 0;
  
  return BSP_Error;
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
