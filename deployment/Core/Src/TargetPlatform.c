/**
  ******************************************************************************
  * @file    TargetPlatform.c
  * @author  System Research & Applications Team - Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   Initialization of the Target Platform
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

#include <stdio.h>
#include "TargetFeatures.h"
#include "main.h"
//#include "sensor_service.h"

#ifdef PREDMNT1_ENABLE_PRINTF
  #include "usbd_core.h"
  #include "usbd_desc.h"
  #include "usbd_cdc.h"
  #include "usbd_cdc_interface.h"
#endif /* PREDMNT1_ENABLE_PRINTF */

/** @addtogroup Projects
  * @{
  */

/** @addtogroup DEMONSTRATIONS Demonstrations
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE Predictive Maintenance BLE
  * @{
  */

/** @addtogroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM Predictive Maintenance Target Platform
  * @{
  */
    
/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_EXPORTED_VARIABLES Predictive Maintenance Target Platform Exported Variables
  * @{
  */

/* Imported variables ---------------------------------------------------------*/
//#ifdef PREDMNT1_ENABLE_PRINTF
//   extern USBD_DescriptorsTypeDef VCP_Desc;
//#endif /* PREDMNT1_ENABLE_PRINTF */

/* Exported variables ---------------------------------------------------------*/
TargetFeatures_t TargetBoardFeatures;

#ifdef PREDMNT1_ENABLE_PRINTF
  USBD_HandleTypeDef  USBD_Device;
#endif /* PREDMNT1_ENABLE_PRINTF */

uint16_t PCM_Buffer[((AUDIO_IN_CHANNELS*AUDIO_IN_SAMPLING_FREQUENCY)/1000)  * N_MS ];

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_PRIVATE_VARIABLES Predictive Maintenance Target Platform Private Variables
  * @{
  */

/* Private variables ---------------------------------------------------------*/
BSP_AUDIO_Init_t MicParams;

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_PRIVATE_FUNCTIONS_PROTOTYPES Predictive Maintenance Target Platform Private Functions prototypes
  * @{
  */

/* Local function prototypes --------------------------------------------------*/
static void ISM330DHC_GPIO_Init(void);
  
static void Init_MEMS_Sensors(void);
static void Init_MEMS_Mics(uint32_t AudioFreq, uint32_t AudioVolume);

static uint32_t GetPage(uint32_t Address);
static uint32_t GetBank(uint32_t Address);

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_PRIVATE_FUNCTIONS Predictive Maintenance Target Platform Private Functions
  * @{
  */

/** Configure pins as
        * Analog
        * Input
        * Output
        * EVENT_OUT
        * EXTI
*/
static void ISM330DHC_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct;

  /* GPIO Ports Clock Enable */
  M_INT2_O_GPIO_CLK_ENABLE();

  /*Configure GPIO pin : PC1 */
  GPIO_InitStruct.Pin = M_INT2_O_PIN;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(M_INT2_O_GPIO_PORT, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  /* Enable and set EXTI Interrupt priority */
  HAL_NVIC_SetPriority(M_INT2_O_EXTI_IRQn, 0x00, 0x00);
  HAL_NVIC_EnableIRQ(M_INT2_O_EXTI_IRQn);
}

/** @brief Initialize all the MEMS1 sensors
 * @param None
 * @retval None
 */
static void Init_MEMS_Sensors(void)
{
  PREDMNT1_PRINTF("\nCode compiled for STWIN board\n\r");
  
   /* Accelero & Gyro initialization */
  if(MOTION_SENSOR_Init(ACCELERO_INSTANCE, MOTION_ACCELERO | MOTION_GYRO)==BSP_ERROR_NONE)
  {
    TargetBoardFeatures.AccSensorIsInit= 1;
    TargetBoardFeatures.GyroSensorIsInit= 1;
    
    PREDMNT1_PRINTF("\tOK Accelero Sensor\n\r");
    PREDMNT1_PRINTF("\tOK Gyroscope Sensor\n\r");
  }
  else
  {
    PREDMNT1_PRINTF("\tError Accelero Sensor\n\r");
    PREDMNT1_PRINTF("\tError Gyroscope Sensor\n\r");
  }
    
  /* Magneto initialization */
  if(MOTION_SENSOR_Init(MAGNETO_INSTANCE, MOTION_MAGNETO)==BSP_ERROR_NONE)
  {
    TargetBoardFeatures.MagSensorIsInit= 1;

    PREDMNT1_PRINTF("\tOK Magneto Sensor\n\r");
  }
  else
  {
    PREDMNT1_PRINTF("\tError Magneto Sensor\n\r");
  }
    
  if(ENV_SENSOR_Init(HUMIDITY_INSTANCE,ENV_TEMPERATURE| ENV_HUMIDITY)==BSP_ERROR_NONE)
  {
    TargetBoardFeatures.TempSensorsIsInit[0]= 1;
    TargetBoardFeatures.HumSensorIsInit= 1;
    TargetBoardFeatures.NumTempSensors++;
    PREDMNT1_PRINTF("\tOK Temperature and Humidity (Sensor1)\n\r");
  }
  else
  {
    PREDMNT1_PRINTF("\tError Temperature and Humidity (Sensor1)\n\r");
  }

  if(ENV_SENSOR_Init(PRESSURE_INSTANCE,ENV_TEMPERATURE| ENV_PRESSURE)==BSP_ERROR_NONE)
  {
    TargetBoardFeatures.TempSensorsIsInit[1]= 1;
    TargetBoardFeatures.PressSensorIsInit= 1;
    TargetBoardFeatures.NumTempSensors++;
    PREDMNT1_PRINTF("\tOK Temperature and Pressure (Sensor2)\n\r");
  }
  else
  {
    PREDMNT1_PRINTF("\tError Temperature and Pressure (Sensor2)\n\r");
  }

  /*  Enable all the sensors */
//  if(TargetBoardFeatures.AccSensorIsInit)
//  {
//    if(MOTION_SENSOR_Enable(ACCELERO_INSTANCE, MOTION_ACCELERO)==BSP_ERROR_NONE)
//      PREDMNT1_PRINTF("\tEnabled Accelero Sensor\n\r");
//  }
//  
//  if(TargetBoardFeatures.GyroSensorIsInit)
//  {
//    if(MOTION_SENSOR_Enable(GYRO_INSTANCE, MOTION_GYRO)==BSP_ERROR_NONE)
//      PREDMNT1_PRINTF("\tEnabled Gyroscope Sensor\n\r");
//  }
//  
//  if(TargetBoardFeatures.MagSensorIsInit)
//  {
//    if(MOTION_SENSOR_Enable(MAGNETO_INSTANCE, MOTION_MAGNETO)==BSP_ERROR_NONE)
//      PREDMNT1_PRINTF("\tEnabled Magneto Sensor\n\r");
//  }
//   
//  if(TargetBoardFeatures.TempSensorsIsInit[0])
//  {
//    if(ENV_SENSOR_Enable(TEMPERATURE_INSTANCE_1, ENV_TEMPERATURE)==BSP_ERROR_NONE)
//      PREDMNT1_PRINTF("\tEnabled Temperature\t(Sensor1)\n\r");
//  }
//  
//  if(TargetBoardFeatures.HumSensorIsInit)
//  {
//    if(ENV_SENSOR_Enable(HUMIDITY_INSTANCE, ENV_HUMIDITY)==BSP_ERROR_NONE)
//      PREDMNT1_PRINTF("\tEnabled Humidity\t(Sensor1)\n\r");
//  }
//     
//  if(TargetBoardFeatures.TempSensorsIsInit[1])
//  {
//    if(ENV_SENSOR_Enable(TEMPERATURE_INSTANCE_2, ENV_TEMPERATURE)==BSP_ERROR_NONE)
//      PREDMNT1_PRINTF("\tEnabled Temperature\t(Sensor2)\n\r");
//  }
//  
//  if(TargetBoardFeatures.PressSensorIsInit)
//  {
//    if(ENV_SENSOR_Enable(PRESSURE_INSTANCE, ENV_PRESSURE)==BSP_ERROR_NONE)
//      PREDMNT1_PRINTF("\tEnabled Pressure\t(Sensor2)\n\r");
//  }
}

/** @brief Initialize all the MEMS's Microphones
 * @param None
 * @retval None
 */
static void Init_MEMS_Mics(uint32_t AudioFreq, uint32_t AudioVolume)
{
  /* Initialize microphone acquisition */  
  MicParams.BitsPerSample = 16;
  MicParams.ChannelsNbr = AUDIO_IN_CHANNELS;
  MicParams.Device = ACTIVE_MICROPHONES_MASK;
  MicParams.SampleRate = AudioFreq;
  MicParams.Volume = AudioVolume;
  
  if( BSP_AUDIO_IN_Init(BSP_AUDIO_IN_INSTANCE, &MicParams) != BSP_ERROR_NONE )
  {
    PREDMNT1_PRINTF("\nError Audio Init\r\n");
    
    while(1) {
      ;
    }
  }
  else
  {
    PREDMNT1_PRINTF("\nOK Audio Init\t(Audio Freq.= %ld)\r\n", AudioFreq);
  }
  
  /* Set the volume level */
  if( BSP_AUDIO_IN_SetVolume(BSP_AUDIO_IN_INSTANCE, AudioVolume) != BSP_ERROR_NONE )
  {
    PREDMNT1_PRINTF("Error Audio Volume\r\n\n");
    
    while(1) {
      ;
    }
  }
  else
  {
    PREDMNT1_PRINTF("OK Audio Volume\t(Volume= %ld)\r\n", AudioVolume);
  }

  /* Number of Microphones */
  TargetBoardFeatures.NumMicSensors=AUDIO_IN_CHANNELS;
}

/**
  * @}
  */

/** @defgroup PREDCTIVE_MAINTENANCE_TARGET_PLATFORM_EXPORTED_FUNCTIONS Predictive Maintenance Target Platform Exported Functions
  * @{
  */

/**
  * @brief  Initialize all the Target platform's Features
  * @param  None
  * @retval None
  */
void InitTargetPlatform(void)
{
  //BSP_USART3_Init();
  //BSP_COM_Init(COM1);

#ifdef PREDMNT1_ENABLE_PRINTF
  /* enable USB power on Pwrctrl CR2 register */
  HAL_PWREx_EnableVddUSB();

  /* Configure the CDC */
  /* Init Device Library */
  USBD_Init(&USBD_Device, &VCP_Desc, 0);
  /* Add Supported Class */
  USBD_RegisterClass(&USBD_Device, USBD_CDC_CLASS);
  /* Add Interface callbacks for AUDIO and CDC Class */
  USBD_CDC_RegisterInterface(&USBD_Device, &USBD_CDC_fops);
  /* Start Device Process */
  USBD_Start(&USBD_Device);
  /* 10 seconds ... for having time to open the Terminal
   * for looking the ALLMEMS1 Initialization phase */
  HAL_Delay(10000);
#endif /* PREDMNT1_ENABLE_PRINTF */
  
  /* Initialize button */
  BSP_PB_Init(BUTTON_KEY, BUTTON_MODE_EXTI);
  BSP_PB_PWR_Init();
  
  /* Initialize LED */
  BSP_LED_Init(LED1);  
  
  /* Initialize the Battery Charger */
  BSP_BC_Init();
  
  /* In order to be able to Read Battery Volt */
  BSP_BC_BatMS_Init();
  
  /* In order to Initialize the GPIO for having the battery Status */
  BSP_BC_Chrg_Init();
  
  //BSP_BC_CmdSend(BATMS_ON);
  
  PREDMNT1_PRINTF("\r\nSTMicroelectronics %s:\r\n"
          "\t%s\r\n"
          "\tVersion %c.%c.%c\r\n"
          "\tSTM32L4R9ZI-STWIN board"
          "\r\n\n",
          PREDMNT1_PACKAGENAME,
          CONFIG_NAME,
          PREDMNT1_VERSION_MAJOR,PREDMNT1_VERSION_MINOR,PREDMNT1_VERSION_PATCH);

  /* Reset all the Target's Features */
  memset(&TargetBoardFeatures, 0, sizeof(TargetFeatures_t));
  /* Discovery and Intialize all the MEMS Target's Features */
  
  ISM330DHC_GPIO_Init();
  
  Init_MEMS_Sensors();
  
  PREDMNT1_PRINTF("\n\r");
}

/** @brief Initialize all the MEMS's Microphones
 * @param None
 * @retval None
 */
void InitMics(uint32_t AudioFreq, uint32_t AudioVolume)
{
  Init_MEMS_Mics(AudioFreq, AudioVolume);
   
  BSP_AUDIO_IN_Record(BSP_AUDIO_IN_INSTANCE, (uint8_t *) PCM_Buffer, DEFAULT_AUDIO_IN_BUFFER_SIZE);
}

/** @brief DeInitialize all the MEMS's Microphones
 * @param None
 * @retval None
 */
void DeInitMics(void)
{
  if( BSP_AUDIO_IN_Stop(BSP_AUDIO_IN_INSTANCE) != BSP_ERROR_NONE )
  {
    PREDMNT1_PRINTF("Error Audio Stop\r\n");
    
    while(1) {
      ;
    }
  }
  else
    PREDMNT1_PRINTF("OK Audio Stop\r\n");
  
  
  if( BSP_AUDIO_IN_DeInit(BSP_AUDIO_IN_INSTANCE) != BSP_ERROR_NONE )
  {
    PREDMNT1_PRINTF("Error Audio DeInit\r\n");
    
    while(1) {
      ;
    }
  }
  else
    PREDMNT1_PRINTF("OK Audio DeInit\r\n");
}


/**
  * @brief  This function switches on the LED
  * @param  None
  * @retval None
  */
void LedOnTargetPlatform(void)
{
  BSP_LED_On(LED1);
  TargetBoardFeatures.LedStatus=1;
}

/**
  * @brief  This function switches off the LED
  * @param  None
  * @retval None
  */
void LedOffTargetPlatform(void)
{
  BSP_LED_Off(LED1);
  TargetBoardFeatures.LedStatus=0;
}

/** @brief  This function toggles the LED
  * @param  None
  * @retval None
  */
void LedToggleTargetPlatform(void)
{
  BSP_LED_Toggle(LED2);
}

/**
  * @brief  Gets the page of a given address
  * @param  Addr: Address of the FLASH Memory
  * @retval The page of a given address
  */
static uint32_t GetPage(uint32_t Addr)
{
  uint32_t page = 0;
  
  if (Addr < (FLASH_BASE + FLASH_BANK_SIZE))
  {
    /* Bank 1 */
    page = (Addr - FLASH_BASE) / FLASH_PAGE_SIZE;
  }
  else
  {
    /* Bank 2 */
    page = (Addr - (FLASH_BASE + FLASH_BANK_SIZE)) / FLASH_PAGE_SIZE;
  }
  
  return page;
}

/**
  * @brief  Gets the bank of a given address
  * @param  Addr: Address of the FLASH Memory
  * @retval The bank of a given address
  */
static uint32_t GetBank(uint32_t Addr)
{
  uint32_t bank = 0;
  
  if (READ_BIT(SYSCFG->MEMRMP, SYSCFG_MEMRMP_FB_MODE) == 0)
  {
  	/* No Bank swap */
    if (Addr < (FLASH_BASE + FLASH_BANK_SIZE))
    {
      bank = FLASH_BANK_1;
    }
    else
    {
      bank = FLASH_BANK_2;
    }
  }
  else
  {
  	/* Bank swap */
    if (Addr < (FLASH_BASE + FLASH_BANK_SIZE))
    {
      bank = FLASH_BANK_2;
    }
    else
    {
      bank = FLASH_BANK_1;
    }
  }
  
  return bank;
}

/**
 * @brief User function for Erasing the MDM on Flash
 * @param None
 * @retval uint32_t Success/NotSuccess [1/0]
 */
uint32_t UserFunctionForErasingFlash(void) {
  FLASH_EraseInitTypeDef EraseInitStruct;
  uint32_t SectorError = 0;
  uint32_t Success=1;

  EraseInitStruct.TypeErase   = FLASH_TYPEERASE_PAGES;
  EraseInitStruct.Banks       = GetBank(MDM_FLASH_ADD);
  EraseInitStruct.Page        = GetPage(MDM_FLASH_ADD);
#ifndef STM32L4R9xx
  EraseInitStruct.NbPages     = 2; /* Each page is 2K */
#else /* STM32L4R9xx */
  EraseInitStruct.NbPages     = 1; /* Each page is 4k */
#endif /* STM32L4R9xx */

  /* Unlock the Flash to enable the flash control register access *************/
  HAL_FLASH_Unlock();
  
#ifdef STM32L4R9xx
   /* Clear OPTVERR bit set on virgin samples */
  __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_OPTVERR);
  /* Clear PEMPTY bit set (as the code is executed from Flash which is not empty) */
  if (__HAL_FLASH_GET_FLAG(FLASH_FLAG_PEMPTY) != 0) {
    __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_PEMPTY);
  }
#endif /* STM32L4R9xx */

  if(HAL_FLASHEx_Erase(&EraseInitStruct, &SectorError) != HAL_OK){
    /* Error occurred while sector erase. 
      User can add here some code to deal with this error. 
      SectorError will contain the faulty sector and then to know the code error on this sector,
      user can call function 'HAL_FLASH_GetError()'
      FLASH_ErrorTypeDef errorcode = HAL_FLASH_GetError(); */
    Success=0;
    Error_Handler();
  }

  /* Lock the Flash to disable the flash control register access (recommended
  to protect the FLASH memory against possible unwanted operation) *********/
  HAL_FLASH_Lock();

  return Success;
}

/**
 * @brief User function for Saving the MDM  on the Flash
 * @param void *InitMetaDataVector Pointer to the MDM beginning
 * @param void *EndMetaDataVector Pointer to the MDM end
 * @retval uint32_t Success/NotSuccess [1/0]
 */
uint32_t UserFunctionForSavingFlash(void *InitMetaDataVector,void *EndMetaDataVector)
{
  uint32_t Success=1;

  /* Store in Flash Memory */
  uint32_t Address = MDM_FLASH_ADD;
  uint64_t *WriteIndex;

  /* Unlock the Flash to enable the flash control register access *************/
  HAL_FLASH_Unlock();
  for(WriteIndex =((uint64_t *) InitMetaDataVector); WriteIndex<((uint64_t *) EndMetaDataVector); WriteIndex++) {
    if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD, Address,*WriteIndex) == HAL_OK){
      Address = Address + 8;
    } else {
      /* Error occurred while writing data in Flash memory.
         User can add here some code to deal with this error
         FLASH_ErrorTypeDef errorcode = HAL_FLASH_GetError(); */
      Error_Handler();
      Success =0;
    }
  }

  /* Lock the Flash to disable the flash control register access (recommended
   to protect the FLASH memory against possible unwanted operation) *********/
  HAL_FLASH_Lock();
 
  return Success;
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

/******************* (C) COPYRIGHT 2021 STMicroelectronics *****END OF FILE****/
