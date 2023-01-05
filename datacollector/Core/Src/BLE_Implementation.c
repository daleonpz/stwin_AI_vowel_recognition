/**
  ******************************************************************************
  * @file    BLE_Implementation.c
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V2.4.0
  * @date    07-June-2021
  * @brief   BLE Implementation template file.
  *          This file should be copied to the application folder and renamed
  *          to BLE_Implementation.c.
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
#include "BLE_Manager.h"
#include "OTA.h"

/* Exported Variables --------------------------------------------------------*/
uint8_t connected= FALSE;
int32_t  NeedToClearSecureDB=0;

/* Private variables ------------------------------------------------------------*/
volatile uint32_t FeatureMask;
static uint16_t BLE_ConnectionHandle = 0;
static uint32_t SizeOfUpdateBlueFW=0;
static uint8_t VibrParam[80];
      
/* Private functions ---------------------------------------------------------*/
static uint32_t DebugConsoleParsing(uint8_t * att_data, uint8_t data_length);
static uint32_t ConfigCommandParsing(uint8_t * att_data, uint8_t data_length);
//static void ReadRequestEnvFunction(void);
static void DisconnectionCompletedFunction(void);
static void ConnectionCompletedFunction(uint16_t ConnectionHandle);

static uint32_t DebugConsoleCommandParsing(uint8_t * att_data, uint8_t data_length);
static uint8_t VibrationParametersCommandParsing(uint8_t CommandLenght);

/**********************************************************************************************
 * Callback functions prototypes to manage the extended configuration characteristic commands *
 **********************************************************************************************/
static void ExtExtConfigUidCommandCallback(uint8_t **UID);
static void ExtConfigInfoCommandCallback(uint8_t *Answer);
static void ExtConfigHelpCommandCallback(uint8_t *Answer);
static void ExtConfigVersionFwCommandCallback(uint8_t *Answer);
static void ExtConfigPowerStatusCommandCallback(uint8_t *Answer);

static void ExtConfigSetNameCommandCallback(uint8_t *NewName);
static void ExtConfigReadCustomCommandsCallback(JSON_Array *JSON_SensorArray);
static void ExtConfigCustomCommandCallback(BLE_CustomCommadResult_t *CustomCommand);

/** @brief Initialize the BlueNRG stack and services
  * @param  None
  * @retval None
  */
void BluetoothInit(void)
{
  /* BlueNRG stack setting */
  BlueNRG_StackValue.ConfigValueOffsets                   = CONFIG_DATA_PUBADDR_OFFSET;
  BlueNRG_StackValue.ConfigValuelength                    = CONFIG_DATA_PUBADDR_LEN;
  BlueNRG_StackValue.GAP_Roles                            = GAP_PERIPHERAL_ROLE;
  BlueNRG_StackValue.IO_capabilities                      = IO_CAP_DISPLAY_ONLY;
  BlueNRG_StackValue.AuthenticationRequirements           = BONDING;
  BlueNRG_StackValue.MITM_ProtectionRequirements          = MITM_PROTECTION_REQUIRED;
  BlueNRG_StackValue.SecureConnectionSupportOptionCode    = SC_IS_SUPPORTED;
  BlueNRG_StackValue.SecureConnectionKeypressNotification = KEYPRESS_IS_NOT_SUPPORTED;
  
  /* To set the TX power level of the bluetooth device ( -2,1 dBm )*/
  BlueNRG_StackValue.EnableHighPowerMode= 1; /*  High Power */
  
  /* Values: 0x00 ... 0x31 - The value depends on the device */
  BlueNRG_StackValue.PowerAmplifierOutputLevel =4;
  
  /* BlueNRG services setting */
  BlueNRG_StackValue.EnableConfig    = 0;
  BlueNRG_StackValue.EnableConsole   = 1;
  BlueNRG_StackValue.EnableExtConfig = 1;
  
  /* For Enabling the Secure Connection */
  BlueNRG_StackValue.EnableSecureConnection=0;
  /* Default Secure PIN */
  BlueNRG_StackValue.SecurePIN=123456;
  
  /* For creating a Random Secure PIN */
#ifdef BLE_MANAGER_PRINTF
  BlueNRG_StackValue.EnableRandomSecurePIN = 1;
#else /* BLE_MANAGER_PRINTF */
  BlueNRG_StackValue.EnableRandomSecurePIN = 0;
#endif /* BLE_MANAGER_PRINTF */
  
  BlueNRG_StackValue.AdvertisingFilter    = NO_WHITE_LIST_USE;
  
  if(BlueNRG_StackValue.EnableSecureConnection) {
    /* Using the Secure Connection, the Rescan should be done by BLE chip */    
    BlueNRG_StackValue.ForceRescan =0;
  } else {
    BlueNRG_StackValue.ForceRescan =1;
  }
  
  InitBleManager();
}

/**
 * @brief  Custom Service Initialization.
 * @param  None
 * @retval None
 */
void BLE_InitCustomService(void) {
  /* Define Custom Function for Debug Console Command parsing */
  CustomDebugConsoleParsingCallback = &DebugConsoleParsing;
  
  /* Define Custom Function for Connection Completed */
  CustomConnectionCompleted = &ConnectionCompletedFunction;
  
  /* Define Custom Function for Disconnection Completed */
  CustomDisconnectionCompleted = &DisconnectionCompletedFunction;
  
  /***********************************************************************************
   * Callback functions to manage the extended configuration characteristic commands *
   ***********************************************************************************/
  CustomExtConfigUidCommandCallback  = &ExtExtConfigUidCommandCallback;
  CustomExtConfigInfoCommandCallback = &ExtConfigInfoCommandCallback;
  CustomExtConfigHelpCommandCallback = &ExtConfigHelpCommandCallback;
  CustomExtConfigVersionFwCommandCallback = &ExtConfigVersionFwCommandCallback;
  CustomExtConfigPowerStatusCommandCallback = &ExtConfigPowerStatusCommandCallback;
  
  CustomExtConfigSetNameCommandCallback = &ExtConfigSetNameCommandCallback;
  CustomExtConfigReadCustomCommandsCallback = &ExtConfigReadCustomCommandsCallback;
  CustomExtConfigCustomCommandCallback = &ExtConfigCustomCommandCallback;
  
  /**
  * For each features, user can assign here the pointer at the function for the read request data.
  * For example for the environmental features:
  * 
  * CustomReadRequestEnvFunctionPointer = &ReadRequestEnvFunction;
  * 
  * User can define and insert in the BLE_Implementation.c source code the functions for the read request data
  * ReadRequestEnvFunction function is already defined.
  *
  */
  
//  /* Define Custom Function for Read Request Environmental Data */
//  CustomReadRequestEnvFunctionPointer = &ReadRequestEnvFunction;
  
  /*******************
   * User code begin *
   *******************/
  
  /**
  * User can added here the custom service initialization for the selected BLE features.
  * For example for the environmental features:
  * 
  * //BLE_InitEnvService(PressEnable,HumEnable,NumTempEnabled)
  * BleManagerAddChar(BleCharPointer= BLE_InitEnvService(1, 1, 1));
  */
  
  /* Service initialization and adding for the environmental features */
  /* BLE_InitEnvService(PressEnable,HumEnable,NumTempEnabled) */
  BleManagerAddChar(BLE_InitEnvService(1, 1, 1));
  
  /* Service initialization and adding  for the inertial features */
  /* BLE_InitInertialService(AccEnable,GyroEnable,MagEnabled) */
  BleManagerAddChar(BLE_InitInertialService(1,1,1));
  
  /* Custom service initialization for the audio level features */
  BleManagerAddChar(BLE_InitAudioLevelService(AUDIO_IN_CHANNELS));
  
  /* Service initialization and adding for the battery features */
  BleManagerAddChar(BLE_InitBatteryService());
  
  /* Service initialization and adding for the FFT Amplitude features */
  BleManagerAddChar(BLE_InitFFTAmplitudeService());
  
  /* Service initialization and adding for the Time Domain features */
  BleManagerAddChar(BLE_InitTimeDomainService());
  
  /* Service initialization and adding for the FFT Alarm Speed Status features */
  BleManagerAddChar(BLE_InitFFTAlarmSpeedStatusService());
  
  /* Service initialization and adding for the FFT Alarm Acc Peak Status features */
  BleManagerAddChar(BLE_InitFFTAlarmAccPeakStatusService());
  
  /* Service initialization and adding for the FFT Alarm Acc Peak Status features */
  BleManagerAddChar(BLE_InitFFTAlarmSubrangeStatusService());
  
  /*****************
   * User code end *
   *****************/
}

/**
 * @brief  Set Custom Advertize Data.
 * @param  uint8_t *manuf_data: Advertize Data
 * @retval None
 */
void BLE_SetCustomAdvertizeData(uint8_t *manuf_data)
{
  /* Identify the used hardware platform  */
  manuf_data[15] = BLE_MANAGER_USED_PLATFORM;
  
  /**
  * User can add here the custom advertize data setting  for the selected BLE features.
  * For example for the environmental features:
  * 
  * BLE_SetCustomEnvAdvertizeData(manuf_data);
  */
  
  /* Custom advertize data setting for the environmental features */
  BLE_SetEnvAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the inertial features */
  BLE_SetInertialAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the audio level features */
  BLE_SetAudioLevelAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the battery features */
  BLE_SetBatteryAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the FFT Amplitude features */
  BLE_SetFFTAmplitudeAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the Time Domain features */
  BLE_SetTimeDomainAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the FFT Alarm Speed Status features */
  BLE_SetFFTAlarmSpeedStatusAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the FFT Alarm Acc Peak Status features */
  BLE_SetFFTAlarmAccPeakStatusAdvertizeData(manuf_data);
  
  /* Custom advertize data setting for the FFT Alarm Acc Peak Status features */
  BLE_SetFFTAlarmSubrangeStatusAdvertizeData(manuf_data);

  /* Adds BLE MAC address in advertize data */
  manuf_data[20] = BlueNRG_StackValue.BleMacAddress[5];
  manuf_data[21] = BlueNRG_StackValue.BleMacAddress[4];
  manuf_data[22] = BlueNRG_StackValue.BleMacAddress[3];
  manuf_data[23] = BlueNRG_StackValue.BleMacAddress[2];
  manuf_data[24] = BlueNRG_StackValue.BleMacAddress[1];
  manuf_data[25] = BlueNRG_StackValue.BleMacAddress[0];
  
}

/**
 * @brief  This function makes the parsing of the Configuration Commands
 * @param uint8_t *att_data attribute data
 * @param uint8_t data_length length of the data
 * @retval uint32_t SendItBack true/false
 */
static uint32_t ConfigCommandParsing(uint8_t * att_data, uint8_t data_length)
{
  FeatureMask = (att_data[3]) | (att_data[2]<<8) | (att_data[1]<<16) | (att_data[0]<<24);
  uint8_t Command = att_data[4];
  uint8_t Data    = att_data[5];
  uint32_t SendItBack = 1;

  switch (FeatureMask) {
    /* Environmental features */
    case FEATURE_MASK_TEMP1:
    case FEATURE_MASK_TEMP2:
    case FEATURE_MASK_PRESS:
    case FEATURE_MASK_HUM:
      switch(Command) {
        case 255:
          /* Change the Sending interval */
          if(Data!=0) {
            /* Multiple of 100mS */
            uhCCR1_Val  = 1000*Data;
          } else {
            /* Default Value */
            uhCCR1_Val  = DEFAULT_uhCCR1_Val;
          }
          SendItBack = 0;
        break;
      }
    break;
    /* Inertial features */
    case FEATURE_MASK_ACC:
    case FEATURE_MASK_GRYO:
    case FEATURE_MASK_MAG:
      switch(Command) {
        case 255:
          /* Change the Sending interval */
          if(Data!=0) {
            /* Multiple of 100mS */
            uhCCR3_Val  = 1000*Data;
          } else {
            /* Default Value */
            uhCCR3_Val  = DEFAULT_uhCCR3_Val;
          }
          SendItBack = 0;
        break;
      }
    break;
    /* Mic features */
    case FEATURE_MASK_MIC:
      switch(Command) {
        case 255:
          /* Change the Sending interval */
          if(Data!=0) {
            /* Multiple of 100mS */
            uhCCR2_Val  = 1000*Data;
          } else {
            /* Default Value */
            uhCCR2_Val  = DEFAULT_uhCCR2_Val;
          }
          SendItBack = 0;
        break;
      }
    break;
  }
  
#ifdef PREDMNT1_DEBUG_CONNECTION
  if(!SendItBack)
  {
    if(BLE_StdTerm_Service==BLE_SERV_ENABLE) {
      BytesToWrite = sprintf((char *)BufferToWrite,"Conf Sig F=%lx C=%2x Data=%2x\n\r",FeatureMask,Command,Data);
      Term_Update(BufferToWrite,BytesToWrite);
    } else {
      PREDMNT1_PRINTF("Conf Sig F=%lx C=%2x Data=%2x\n\r",FeatureMask,Command,Data);
    }
  }
#endif /* PREDMNT1_DEBUG_CONNECTION */
  return SendItBack;
}

/**
* @brief  This function makes the parsing of the Debug Console
* @param  uint8_t *att_data attribute data
* @param  uint8_t data_length length of the data
* @retval uint32_t SendBackData true/false
*/
static uint32_t DebugConsoleParsing(uint8_t * att_data, uint8_t data_length)
{
  /* By default Answer with the same message received */
  uint32_t SendBackData =1; 
  
  if(SizeOfUpdateBlueFW!=0) {
    /* FP-IND-PREDMNT1 firwmare update */
    int8_t RetValue = UpdateFWBlueMS(&SizeOfUpdateBlueFW,att_data, data_length,1);
    if(RetValue!=0) {
      Term_Update((uint8_t *)&RetValue,1);
      if(RetValue==1) {
        /* if OTA checked */
        //BytesToWrite =sprintf((char *)BufferToWrite,"The Board will restart in 5 seconds\r\n");
        //Term_Update(BufferToWrite,BytesToWrite);
        PREDMNT1_PRINTF("%s will restart in 5 seconds\r\n",PREDMNT1_PACKAGENAME);
        HAL_Delay(5000);
        HAL_NVIC_SystemReset();
      }
    }
    SendBackData=0;
  } else {
    /* Received one write from Client on Terminal characteristc */
    SendBackData = DebugConsoleCommandParsing(att_data,data_length);
  }
  
  return SendBackData;
}

/**
 * @brief  This function makes the parsing of the Debug Console Commands
 * @param  uint8_t *att_data attribute data
 * @param  uint8_t data_length length of the data
 * @retval uint32_t SendBackData true/false
 */
static uint32_t DebugConsoleCommandParsing(uint8_t * att_data, uint8_t data_length)
{
  uint32_t SendBackData = 1;
  
  static uint8_t SetVibrParam= 0;

    /* Help Command */
    if(!strncmp("help",(char *)(att_data),4)) {
      /* Print Legend */
      SendBackData=0;

      BytesToWrite =sprintf((char *)BufferToWrite,"Command:\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"info -> System Info\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"versionFw  -> FW Version\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"getVibrParam  -> Read Vibration Parameters\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"setVibrParam [-odr -fs -size -wind - tacq -subrng -ovl] -> Set Vibration Parameters");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\nodr= [13, 26, 52, 104, 208, 416, 833, 1660, 3330, 6660]");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\nfs= [2, 4, 8, 16]");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\nsize= [256, 512, 1024, 2048]");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\nwind= [RECTANGULAR= 0, HANNING= 1, HAMMING= 2, FLAT_TOP= 3]");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\ntacq= [500 - 60000]");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\nsubrng= [8, 16, 32, 64]");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\novl= [5 - 95]\r\n\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
      HAL_Delay(20);
      BytesToWrite =sprintf((char *)BufferToWrite,"setName xxxxxxx -> Set the node name (Max 7 characters)\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
    } else if(!strncmp("versionFw",(char *)(att_data),9)) {
      BytesToWrite =sprintf((char *)BufferToWrite,"%s_%s_%c.%c.%c\r\n",
#ifdef STM32F401xE
                            "F401"
#elif STM32F446xx
                            "F446"
#elif STM32L476xx
                            "L476"
#elif STM32L4R9xx
                            "L4R9"
#else
#error "Undefined STM32 processor type"
#endif
                            ,PREDMNT1_PACKAGENAME,
                            PREDMNT1_VERSION_MAJOR,
                            PREDMNT1_VERSION_MINOR,
                            PREDMNT1_VERSION_PATCH);
#ifdef DISABLE_FOTA
      if(FirstCommandSent)
      {
        FirstCommandSent= 0;
#endif /* DISABLE_FOTA */
        Term_Update(BufferToWrite,BytesToWrite);
        SendBackData=0;
#ifdef DISABLE_FOTA
      }
      else
        SendBackData=1;
#endif /* DISABLE_FOTA */
    } else if(!strncmp("info",(char *)(att_data),4)) {
      SendBackData=0;
      
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\nSTMicroelectronics %s:\r\n"
          "\tVersion %c.%c.%c\r\n"
          "\tSTM32F446xx-Nucleo board"
          "\r\n",
          PREDMNT1_PACKAGENAME,
          PREDMNT1_VERSION_MAJOR,PREDMNT1_VERSION_MINOR,PREDMNT1_VERSION_PATCH);
      Term_Update(BufferToWrite,BytesToWrite);

      BytesToWrite =sprintf((char *)BufferToWrite,"\t(HAL %ld.%ld.%ld_%ld)\r\n"
        "\tCompiled %s %s"
#if defined (__IAR_SYSTEMS_ICC__)
        " (IAR)\r\n",
#elif defined (__CC_ARM)
        " (KEIL)\r\n",
#elif defined (__GNUC__)
        " (STM32CubeIDE)\r\n",
#endif
          HAL_GetHalVersion() >>24,
          (HAL_GetHalVersion() >>16)&0xFF,
          (HAL_GetHalVersion() >> 8)&0xFF,
           HAL_GetHalVersion()      &0xFF,
           __DATE__,__TIME__);
      Term_Update(BufferToWrite,BytesToWrite);
      
    BytesToWrite =sprintf((char *)BufferToWrite,"Code compiled for STWIN board\r\n");
    
    Term_Update(BufferToWrite,BytesToWrite);
      
#ifndef DISABLE_FOTA
    }  if(!strncmp("upgradeFw",(char *)(att_data),9)) {
      uint32_t uwCRCValue;
      uint8_t *PointerByte = (uint8_t*) &SizeOfUpdateBlueFW;

      SizeOfUpdateBlueFW=atoi((char *)(att_data+9));
      PointerByte[0]=att_data[ 9];
      PointerByte[1]=att_data[10];
      PointerByte[2]=att_data[11];
      PointerByte[3]=att_data[12];

      /* Check the Maximum Possible OTA size */
      if(SizeOfUpdateBlueFW>OTA_MAX_PROG_SIZE) {
        PREDMNT1_PRINTF("OTA %s SIZE=%ld > %d Max Allowed\r\n",PREDMNT1_PACKAGENAME,SizeOfUpdateBlueFW, OTA_MAX_PROG_SIZE);
        /* UserAnswer with a wrong CRC value for signaling the problem to BlueMS application */
        PointerByte[0]= att_data[13];
        PointerByte[1]=(att_data[14]!=0) ? 0 : 1;/* In order to be sure to have a wrong CRC */
        PointerByte[2]= att_data[15];
        PointerByte[3]= att_data[16];
        BytesToWrite = 4;
        Term_Update(BufferToWrite,BytesToWrite);
      } else {
        PointerByte = (uint8_t*) &uwCRCValue;
        PointerByte[0]=att_data[13];
        PointerByte[1]=att_data[14];
        PointerByte[2]=att_data[15];
        PointerByte[3]=att_data[16];

        PREDMNT1_PRINTF("OTA %s SIZE=%ld uwCRCValue=%lx\r\n",PREDMNT1_PACKAGENAME,SizeOfUpdateBlueFW,uwCRCValue);
	  
        /* Reset the Flash */
        StartUpdateFWBlueMS(SizeOfUpdateBlueFW,uwCRCValue);

        /* Reduce the connection interval */
        {
          int ret = aci_l2cap_connection_parameter_update_req(BLE_ConnectionHandle,
                                                        10 /* interval_min*/,
                                                        10 /* interval_max */,
                                                        0   /* slave_latency */,
                                                        400 /*timeout_multiplier*/);
          /* Go to infinite loop if there is one error */
          if (ret != BLE_STATUS_SUCCESS) {
            while (1) {
              PREDMNT1_PRINTF("Problem Changing the connection interval\r\n");
            }
          }
        }
        
        /* Signal that we are ready sending back the CRV value*/
        BufferToWrite[0] = PointerByte[0];
        BufferToWrite[1] = PointerByte[1];
        BufferToWrite[2] = PointerByte[2];
        BufferToWrite[3] = PointerByte[3];
        BytesToWrite = 4;
        Term_Update(BufferToWrite,BytesToWrite);
      }
      
      SendBackData=0;      
    } else if(!strncmp("versionBle",(char *)(att_data),10)) {
      uint8_t  hwVersion;
      uint16_t fwVersion;
      /* get the BlueNRG HW and FW versions */
      getBlueNRG2_Version(&hwVersion, &fwVersion);
      BytesToWrite =sprintf((char *)BufferToWrite,"%s_v%d.%d.%c\r\n",
                            "BlueNRG2",
                            (fwVersion>>8)&0xF,
                            (fwVersion>>4)&0xF,
                            ('a' + (fwVersion&0xF)));
      Term_Update(BufferToWrite,BytesToWrite);
      SendBackData=0; 
    } else if(!strncmp("getVibrParam",(char *)(att_data),12)) {
      BytesToWrite =sprintf((char *)BufferToWrite,"\r\nAccelerometer parameters:\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
      BytesToWrite =sprintf((char *)BufferToWrite,"AccFifoBdr= %d fs= %d\r\n",
                            AcceleroParams.AccFifoBdr,
                            AcceleroParams.fs);
      Term_Update(BufferToWrite,BytesToWrite);
      
      BytesToWrite =sprintf((char *)BufferToWrite,"MotionSP parameters:\r\n");
      Term_Update(BufferToWrite,BytesToWrite);
      BytesToWrite =sprintf((char *)BufferToWrite,"size= %d wind= %d tacq= %d subrng= %d ovl= %d\r\n",
                            MotionSP_Parameters.FftSize,
                            MotionSP_Parameters.window,
                            MotionSP_Parameters.tacq,
                            MotionSP_Parameters.subrange_num,
                            MotionSP_Parameters.FftOvl);
      Term_Update(BufferToWrite,BytesToWrite);
      SendBackData=0;
    } else if(!strncmp("setVibrParam",(char *)(att_data),12)) {
      SetVibrParam= 1;
      SendBackData=0;
#endif /* DISABLE_FOTA */
    } else if((att_data[0]=='u') & (att_data[1]=='i') & (att_data[2]=='d')) {
      /* Write back the STM32 UID */
      uint8_t *uid = (uint8_t *)STM32_UUID;
      uint32_t MCU_ID = STM32_MCU_ID[0]&0xFFF;
      BytesToWrite =sprintf((char *)BufferToWrite,"%.2X%.2X%.2X%.2X%.2X%.2X%.2X%.2X%.2X%.2X%.2X%.2X_%.3lX\r\n",
                            uid[ 3],uid[ 2],uid[ 1],uid[ 0],
                            uid[ 7],uid[ 6],uid[ 5],uid[ 4],
                            uid[11],uid[ 10],uid[9],uid[8],
                            MCU_ID);
      Term_Update(BufferToWrite,BytesToWrite);
      SendBackData=0;
    } else if(!strncmp("setName ",(char *)(att_data),8)) {
      
      //int NameLength= strcspn((const char *)att_data,"\n");
      int NameLength= data_length -1;
      
      if(NameLength > 8)
      {
        for(int i=1;i<8;i++)
          NodeName[i]= atoi(" ");
 
        if((NameLength - 8) > 7)
          NameLength= 7;
        else NameLength= NameLength - 8;
        
        for(int i=1;i<NameLength+1;i++)
        {
          NodeName[i]= att_data[i+7];
          BlueNRG_StackValue.BoardName[i-1]= att_data[i+7];
        }
        
        MDM_SaveGMD(GMD_NODE_NAME,(void *)&NodeName);
        NecessityToSaveMetaDataManager=1;
        
        BytesToWrite =sprintf((char *)BufferToWrite,"\nThe node nome has been updated\r\n");
        Term_Update(BufferToWrite,BytesToWrite);
        BytesToWrite =sprintf((char *)BufferToWrite,"Disconnecting and riconnecting to see the new node name\r\n");
        Term_Update(BufferToWrite,BytesToWrite);
      }
      else
      {
        BytesToWrite =sprintf((char *)BufferToWrite,"\nInsert the node name\r\n");
        Term_Update(BufferToWrite,BytesToWrite);
        BytesToWrite =sprintf((char *)BufferToWrite,"Use command: setName 'xxxxxxx'\r\n");
        Term_Update(BufferToWrite,BytesToWrite);
      }

      SendBackData=0;
    }

    if(SetVibrParam)
    {
      uint8_t Index=0;
     
      static uint8_t NumByte= 0;
      static uint8_t CommandLenght=0;
      
      while( (att_data[Index] != '\n') && (att_data[Index] != '\0') )
      {
        VibrParam[20*NumByte + Index]= att_data[Index];
        Index++;
        CommandLenght++;
      }
      
      NumByte++;
      
      if(att_data[Index] == '\n')
      {
        if(VibrationParametersCommandParsing(CommandLenght))
        {
          /* Save vibration parameters values to memory */
          SaveVibrationParamToMemory();
        }
        
        NumByte= 0;
        SetVibrParam=0;
        CommandLenght=0;
      }
      
      SendBackData= 0;
    }
    
#if 1

  /* If it's something not yet recognized... only for testing.. This must be removed*/
   if(SendBackData) {
    if(att_data[0]=='@') {

      if(att_data[1]=='T') {
        uint8_t loc_att_data[6];
        uint8_t loc_data_length=6;

        loc_att_data[0] = (FEATURE_MASK_TEMP1>>24)&0xFF;
        loc_att_data[1] = (FEATURE_MASK_TEMP1>>16)&0xFF;
        loc_att_data[2] = (FEATURE_MASK_TEMP1>>8 )&0xFF;
        loc_att_data[3] = (FEATURE_MASK_TEMP1    )&0xFF;
        loc_att_data[4] = 255;
        
        switch(att_data[2]) {
          case 'L':
            loc_att_data[5] = 50; /* @5S */
          break;
          case 'M':
            loc_att_data[5] = 10; /* @1S */
          break;
          case 'H':
            loc_att_data[5] = 1; /* @100mS */
          break;
          case 'D':
            loc_att_data[5] = 0; /* Default */
          break;
        }
        SendBackData = ConfigCommandParsing(loc_att_data,loc_data_length);
      } else if(att_data[1]=='A') {
        uint8_t loc_att_data[6];
        uint8_t loc_data_length=6;

        loc_att_data[0] = (FEATURE_MASK_ACC>>24)&0xFF;
        loc_att_data[1] = (FEATURE_MASK_ACC>>16)&0xFF;
        loc_att_data[2] = (FEATURE_MASK_ACC>>8 )&0xFF;
        loc_att_data[3] = (FEATURE_MASK_ACC    )&0xFF;
        loc_att_data[4] = 255;

        switch(att_data[2]) {
          case 'L':
            loc_att_data[5] = 50; /* @5S */
          break;
          case 'M':
            loc_att_data[5] = 10; /* @1S */
          break;
          case 'H':
            loc_att_data[5] = 1; /* @100mS */
          break;
          case 'D':
            loc_att_data[5] = 0; /* Default */
          break;
        }
        SendBackData = ConfigCommandParsing(loc_att_data,loc_data_length);
      } else if(att_data[1]=='M') {
        uint8_t loc_att_data[6];
        uint8_t loc_data_length=6;

        loc_att_data[0] = (FEATURE_MASK_MIC>>24)&0xFF;
        loc_att_data[1] = (FEATURE_MASK_MIC>>16)&0xFF;
        loc_att_data[2] = (FEATURE_MASK_MIC>>8 )&0xFF;
        loc_att_data[3] = (FEATURE_MASK_MIC    )&0xFF;
        loc_att_data[4] = 255;

        switch(att_data[2]) {
          case 'L':
            loc_att_data[5] = 50; /* @5S */
          break;
          case 'M':
            loc_att_data[5] = 10; /* @1S */
          break;
          case 'H':
            loc_att_data[5] = 1; /* @100mS */
          break;
          case 'D':
            loc_att_data[5] = 0; /* Default */
          break;
        }
        SendBackData = ConfigCommandParsing(loc_att_data,loc_data_length);
      }    
    }
  }
#endif
  
  return SendBackData;
}

/**
 * @brief  This function makes the parsing of the set Vibration Parameter Commands
 * @param uint8_t CommandLenght length of the data
 * @retval UpdatedParameters
 */
static uint8_t VibrationParametersCommandParsing(uint8_t CommandLenght)
{
  uint8_t UpdatedParameters= 0;
  uint8_t UpdatedAccParameters= 0;
  
  uint8_t i=7;
  uint32_t Param[7];
  uint8_t DigitNumber;
  uint8_t ParamFound;
  
  int Index= 13;
  
  if(Index >= CommandLenght)
  {
    BytesToWrite =sprintf((char *)BufferToWrite,"\r\nParameters not found\r\n");
    Term_Update(BufferToWrite,BytesToWrite);
  }  
  
  while(Index < CommandLenght)
  {
    Index++;
    ParamFound= 0;
    
    if((VibrParam[Index]=='o') & (VibrParam[Index+1]=='d') & (VibrParam[Index+2]=='r'))
    {
      Index+= 4;
      i=0;
      ParamFound= 1;
    }
    
    if((VibrParam[Index]=='f') & (VibrParam[Index+1]=='s'))
    {
      Index+= 3;
      i=1;
      ParamFound= 1;
    }
    
    if((VibrParam[Index]=='s') & (VibrParam[Index+1]=='i') & (VibrParam[Index+2]=='z') & (VibrParam[Index+3]=='e'))
    {
      Index+= 5;
      i=2;
      ParamFound= 1;
    }
    
    if((VibrParam[Index]=='w') & (VibrParam[Index+1]=='i') & (VibrParam[Index+2]=='n') & (VibrParam[Index+3]=='d'))
    {
      Index+= 5;
      i=3;
      ParamFound= 1;
    }
    
    if((VibrParam[Index]=='t') & (VibrParam[Index+1]=='a') & (VibrParam[Index+2]=='c') & (VibrParam[Index+3]=='q'))
    {
      Index+= 5;
      i=4;
      ParamFound= 1;
    }
    
    if((VibrParam[Index]=='o') & (VibrParam[Index+1]=='v') & (VibrParam[Index+2]=='l') )
    {
      Index+= 4;
      i=5;
      ParamFound= 1;
    }
    
    if((VibrParam[Index]=='s') & (VibrParam[Index+1]=='u') & (VibrParam[Index+2]=='b') & (VibrParam[Index+3]=='r') & (VibrParam[Index+4]=='n') & (VibrParam[Index+5]=='g'))
    {
      Index+= 7;
      i=6;
      ParamFound= 1;
    }
      
    if(ParamFound == 1)
    {
      ParamFound= 0;
      
      DigitNumber= 0;
      while( (VibrParam[Index + DigitNumber] != ' ') &&
             (VibrParam[Index + DigitNumber] != '\r') &&
             (VibrParam[Index + DigitNumber] != '\0') )
        DigitNumber++;
      
      Param[i]= VibrParam[Index + DigitNumber - 1] & 0x0F;
      
      if(DigitNumber > 1)
      {
        for(int t=1; t<DigitNumber; t++)
        {
          Param[i]= Param[i] + ( (VibrParam[Index + DigitNumber - t - 1] & 0x0F) * ((uint32_t)pow(10.0,t)) );
        }           
      }

      switch(i)
      {
      /* AccFifoBdr (FIFO Accelerometer Output Data Rate in Hz) */
      case 0:
        if( (Param[i] == 13)   || (Param[i] == 26)   || (Param[i] == 52)   ||
            (Param[i] == 104)  || (Param[i] == 208)  || (Param[i] == 416)  ||
            (Param[i] == 833)  || (Param[i] == 1660) || (Param[i] == 3330) || (Param[i] == 6660)
           )
        {
          AcceleroParams.AccFifoBdr= Param[i];
          AcceleroParams.AccOdr=  Param[i];
          UpdatedParameters= 1;
          UpdatedAccParameters= 1;
        }
        else
        {
          BytesToWrite =sprintf((char *)BufferToWrite,"\r\nValue out of range for odr\r\n");
          Term_Update(BufferToWrite,BytesToWrite);
        }
        break;
      /* fs (Full Scale in g) */
      case 1:
        if( (Param[i] == 2) || (Param[i] == 4) || (Param[i] == 8) || (Param[i] == 16) )
        {
          AcceleroParams.fs= Param[i];
          UpdatedParameters= 1;
          UpdatedAccParameters= 1;
        }
        else
        {
          BytesToWrite =sprintf((char *)BufferToWrite,"\r\nValue out of range for fs\r\n");
          Term_Update(BufferToWrite,BytesToWrite);
        }
        break;
      /* size (FFT SIZE) */
      case 2:
        if( (Param[i] == 256) || (Param[i] == 512) || (Param[i] == 1024) || (Param[i] == 2048))
        {
          MotionSP_Parameters.FftSize= Param[i];
          UpdatedParameters= 1;
        }          
        else
        {
          BytesToWrite =sprintf((char *)BufferToWrite,"\r\nValue out of range for size\r\n");
          Term_Update(BufferToWrite,BytesToWrite);
        }
        break;
      /*  wind (PRE-FFT WINDOWING Method) */
      case 3:
        if(Param[i] < 4)
        {
          MotionSP_Parameters.window= Param[i];
          UpdatedParameters= 1;
        }          
        else
        {
          BytesToWrite =sprintf((char *)BufferToWrite,"\r\nValue out of range for wind\r\n");
          Term_Update(BufferToWrite,BytesToWrite);
        }
        break;
      /* tacq (TIME ACQUISITION WINDOW in ms) */
      case 4:
        if( (Param[i] >= 500) && (Param[i] <= 60000) )
        {
          MotionSP_Parameters.tacq= Param[i];
          UpdatedParameters= 1;
        }          
        else
        {
          BytesToWrite =sprintf((char *)BufferToWrite,"\r\nValue out of range for tacq\r\n");
          Term_Update(BufferToWrite,BytesToWrite);
        }
        break;
      /* ovl (FFT OVERLAPPING in %) */
      case 5:
        if( (Param[i] >= 5) && (Param[i] <= 95) )
        {
          MotionSP_Parameters.FftOvl= Param[i];
          UpdatedParameters= 1;
        }
        else
        {
          BytesToWrite =sprintf((char *)BufferToWrite,"\r\nValue out of range for ovl\r\n");
          Term_Update(BufferToWrite,BytesToWrite);
        }
        break;
      /*  subrng (SUBRANGE number for evaluate thresholds) */
      case 6:
        if( (Param[i] == 8) || (Param[i] == 16) || (Param[i] == 32) || (Param[i] == 64) )
        {
          MotionSP_Parameters.subrange_num= Param[i];
          UpdatedParameters= 1;
        }
        else
        {
          BytesToWrite =sprintf((char *)BufferToWrite,"\r\nValue out of range for nsubrng\r\n");
          Term_Update(BufferToWrite,BytesToWrite);
        }
        break;
      }
      
      Index= Index + DigitNumber + 1;
    }
    else
    {
      if(VibrParam[Index] != '-')
      {
        BytesToWrite =sprintf((char *)BufferToWrite,"\r\nParam not found\r\n");
        Term_Update(BufferToWrite,BytesToWrite);
      }
    }
  }
  
  BytesToWrite =sprintf((char *)BufferToWrite,"\r\nOK\r\n");
  Term_Update(BufferToWrite,BytesToWrite);  
  
//  BytesToWrite =sprintf((char *)BufferToWrite,"\r\nNew accelerometer parameters:\r\n");
//  Term_Update(BufferToWrite,BytesToWrite);
//  BytesToWrite =sprintf((char *)BufferToWrite,"FifoOdr= %d fs= %d\r\n",
//                        AcceleroParams.FifoOdr,
//                        AcceleroParams.fs);
//  Term_Update(BufferToWrite,BytesToWrite);
//  
//  BytesToWrite =sprintf((char *)BufferToWrite,"New MotionSP parameters:\r\n");
//  Term_Update(BufferToWrite,BytesToWrite);
//  BytesToWrite =sprintf((char *)BufferToWrite,"size= %d wind= %d tacq= %d ovl= %d subrng= %d\r\n",
//                        MotionSP_Parameters.FftSize,
//                        MotionSP_Parameters.window,
//                        MotionSP_Parameters.tacq,
//                        MotionSP_Parameters.FftOvl,
//                        MotionSP_Parameters.subrange_num);
//  Term_Update(BufferToWrite,BytesToWrite);
  
  if(UpdatedAccParameters)
    MotionSP_AcceleroConfig();
  
  return UpdatedParameters;
}


///**
// * @brief  This function is called when there is a Bluetooth Read request.
// * @param  None 
// * @retval None
// */
//static void ReadRequestEnvFunction(void)
//{
//  /* Read Request for Pressure,Humidity, and Temperatures*/
//  int32_t PressToSend;
//  uint16_t HumToSend;
//  int16_t TempToSend;
//
//  /* Read all the Environmental Sensors */
//  ReadEnvironmentalData(&PressToSend,&HumToSend, &TempToSend);
//  
//  /* Send the Data with BLE */
//  BLE_EnvironmentalUpdate(PressToSend,HumToSend,TempToSend, 0);
//  
//  BLE_MANAGER_PRINTF("Read for Env\r\n");
//}

/**
 * @brief  This function is called when the peer device get disconnected.
 * @param  None 
 * @retval None
 */
static void DisconnectionCompletedFunction(void)
{
  connected = FALSE;
  
  /* Disable all timer */
  if(EnvironmentalTimerEnabled) {
    /* Stop the TIM Base generation in interrupt mode (for environmental sensor) */
    if(HAL_TIM_OC_Stop_IT(&TimCCHandle, TIM_CHANNEL_1) != HAL_OK){
      /* Stopping Error */
      Error_Handler();
    }
    
    EnvironmentalTimerEnabled= 0;
  }
  
  if(AudioLevelTimerEnabled) {
    AudioLevelEnable= 0;
    
    /* Stop the TIM Base generation in interrupt mode (for mic audio level) */
    if(HAL_TIM_OC_Stop_IT(&TimCCHandle, TIM_CHANNEL_2) != HAL_OK){
      /* Stopping Error */
      Error_Handler();
    }  
    
    AudioLevelTimerEnabled= 0;
  }
  
  if(InertialTimerEnabled){
    /* Stop the TIM Base generation in interrupt mode (for Acc/Gyro/Mag sensor) */
    if(HAL_TIM_OC_Stop_IT(&TimCCHandle, TIM_CHANNEL_3) != HAL_OK){
      /* Stopping Error */
      Error_Handler();
    }      
    
    InertialTimerEnabled= 0;
  }
  
  if(BatteryTimerEnabled) {
    /* Stop the TIM Base generation in interrupt mode (for battery info) */
    if(HAL_TIM_OC_Stop_IT(&TimCCHandle, TIM_CHANNEL_4) != HAL_OK){
      /* Stopping Error */
      Error_Handler();
    }
    
    BatteryTimerEnabled= 0;
    
    BSP_BC_CmdSend(BATMS_OFF);
  }
  
    if(PredictiveMaintenance)
    {
      disable_FIFO();
      EnableDisable_ACC_HP_Filter(0);
      PredictiveMaintenance= 0;
      FFT_Alarm= 0;
      MotionSP_Running = 0;
    }
      
  /* Reset for any problem during FOTA update */
  SizeOfUpdateBlueFW = 0;
      
  BLE_MANAGER_PRINTF("Call to DisconnectionCompletedFunction\r\n");
  BLE_MANAGER_DELAY(100);
}

/**
 * @brief  This function is called when there is a LE Connection Complete event.
 * @param  None 
 * @retval None
 */
static void ConnectionCompletedFunction(uint16_t ConnectionHandle)
{
  connected = TRUE;
  
  BLE_ConnectionHandle = ConnectionHandle;
  
  BLE_MANAGER_PRINTF("Call to ConnectionCompletedFunction\r\n");
  BLE_MANAGER_DELAY(100);
}

/***********************************************************************************
 * Callback functions to manage the extended configuration characteristic commands *
 ***********************************************************************************/

/**
 * @brief  Callback Function for managing the custom command
 * @param  BLE_CustomCommadResult_t *CustomCommand:
 * @param                            uint8_t *CommandName: Nome of the command
 * @param                            CustomCommand->CommandType: Type of the command
 * @param                            int32_t IntValue:    Integer or boolean parameter
 * @param                            uint8_t *StringValue: String parameter
 * @retval None
 */
static void  ExtConfigCustomCommandCallback(BLE_CustomCommadResult_t *CustomCommand)
{
  BLE_MANAGER_PRINTF("Received Custom Command:\r\n");
  BLE_MANAGER_PRINTF("\tCommand Name: <%s>\r\n", CustomCommand->CommandName);
  BLE_MANAGER_PRINTF("\tCommand Type: <%d>\r\n", CustomCommand->CommandType);
    
  switch(CustomCommand->CommandType) { 
    case BLE_CUSTOM_COMMAND_VOID:
    break;
    case BLE_CUSTOM_COMMAND_INTEGER:
      BLE_MANAGER_PRINTF("\tInt    Value: <%ld>\r\n", CustomCommand->IntValue);
      
      if(!strncmp((char *)CustomCommand->CommandName,"FFT_Overlapping",15))
      {
        MotionSP_Parameters.FftOvl= CustomCommand->IntValue;
        SaveVibrationParamToMemory();
      } else if(!strncmp((char *)CustomCommand->CommandName,"AquisitionTime",14)) {
        MotionSP_Parameters.tacq= CustomCommand->IntValue;
        SaveVibrationParamToMemory();
      }
    break;
    case BLE_CUSTOM_COMMAND_ENUM_INTEGER:
      BLE_MANAGER_PRINTF("\tInt     Enum: <%ld>\r\n", CustomCommand->IntValue);
      
      if(!strncmp((char *)CustomCommand->CommandName,"SensorFullScale",15))
      {
        AcceleroParams.fs= CustomCommand->IntValue;
        SaveVibrationParamToMemory();
      } else if(!strncmp((char *)CustomCommand->CommandName,"SensorOutputDataRate",20)) {
        AcceleroParams.AccOdr= CustomCommand->IntValue;
        AcceleroParams.AccFifoBdr= CustomCommand->IntValue;
        SaveVibrationParamToMemory();
      } else if(!strncmp((char *)CustomCommand->CommandName,"FFT_Size",8)) {
        MotionSP_Parameters.FftSize= CustomCommand->IntValue;
        SaveVibrationParamToMemory();
      } else if(!strncmp((char *)CustomCommand->CommandName,"NumberOfBubrange",16)) {
        MotionSP_Parameters.subrange_num= CustomCommand->IntValue;
        SaveVibrationParamToMemory();
      }
    break;
    case BLE_CUSTOM_COMMAND_BOOLEAN:
      BLE_MANAGER_PRINTF("\tInt    Value: <%ld>\r\n", CustomCommand->IntValue);
    break;
    case  BLE_CUSTOM_COMMAND_STRING:
      BLE_MANAGER_PRINTF("\tString Value: <%s>\r\n", CustomCommand->StringValue);
    break;
    case  BLE_CUSTOM_COMMAND_ENUM_STRING:
      BLE_MANAGER_PRINTF("\tString  Enum: <%s>\r\n", CustomCommand->StringValue);
      
      if(!strncmp((char *)CustomCommand->CommandName,"FFT_WindowType",15))
      {
        if(!strncmp((char *)CustomCommand->StringValue,"Rectangular",11)) {
          MotionSP_Parameters.window= RECTANGULAR;
        } else if(!strncmp((char *)CustomCommand->StringValue,"Hanning",7)) {
          MotionSP_Parameters.window= HANNING;
        } else if(!strncmp((char *)CustomCommand->StringValue,"Hamming",7)) {
          MotionSP_Parameters.window= HAMMING;
        } else if(!strncmp((char *)CustomCommand->StringValue,"FlatTop",7)) {
          MotionSP_Parameters.window= FLAT_TOP;
        }
        
        SaveVibrationParamToMemory();
      }
    break;
  }
  
  /* Insert here the code for managing the received command */
  
  /* ToBeChanged*/

  /* Understand if it's a valid Command or not */
}

/**
 * @brief  Custom commands definition
 * @param  JSON_Array *JSON_SensorArray
 * @retval None
 */
static void ExtConfigReadCustomCommandsCallback(JSON_Array *JSON_SensorArray)
{
  /* Clear the previous Costom Command List */
  ClearCustomCommandsList();

  {
    //The Last value should be BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN
    int32_t ValidIntValues[]={2,4,8,16,BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN};
    if(AddCustomCommand("SensorFullScale", //Name
                        BLE_CUSTOM_COMMAND_ENUM_INTEGER, //Type
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN, //MIN
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN,  //MAX
                        (void *) ValidIntValues, //Enum Array
                        NULL, //Enum String
                        "Example of Enum Integer", //Description
                        JSON_SensorArray)) {
      BLE_MANAGER_PRINTF("Added Command <%s>\r\n","SensorFullScale");
    } else {
      BLE_MANAGER_PRINTF("Error Adding Command <%s>\r\n","SensorFullScale");
      return;
    }
  }
  
  {
    //The Last value should be BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN
    int32_t ValidIntValues[]={13,26,52,104,208,416,833,1660,3330,6660,BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN};
    if(AddCustomCommand("SensorOutputDataRate", //Name
                        BLE_CUSTOM_COMMAND_ENUM_INTEGER, //Type
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN, //MIN
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN,  //MAX
                        (void *) ValidIntValues, //Enum Array
                        NULL, //Enum String
                        "Example of Enum Integer", //Description
                        JSON_SensorArray)) {
      BLE_MANAGER_PRINTF("Added Command <%s>\r\n","SensorOutputDataRate");
    } else {
      BLE_MANAGER_PRINTF("Error Adding Command <%s>\r\n","SensorOutputDataRate");
      return;
    }
  }
  
  {
    char *ValidStringValues[]={"Rectangular", "Hanning","Hamming","FlatTop",NULL};
    if(AddCustomCommand("FFT_WindowType", //Name
                        BLE_CUSTOM_COMMAND_ENUM_STRING, //Type
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN, //MIN
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN,  //MAX
                        (void *)ValidStringValues, //Enum Array
                        NULL, //Enum Int
                        "Example of Enum String", //Description
                        JSON_SensorArray)) {
      BLE_MANAGER_PRINTF("Added Command <%s>\r\n","FFT_WindowType");
    } else {
      BLE_MANAGER_PRINTF("Error Adding Command <%s>\r\n","FFT_WindowType");
      return;
    }
  }
  
  {
    //The Last value should be BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN
    int32_t ValidIntValues[]={256,512,1024,BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN};
    if(AddCustomCommand("FFT_Size", //Name
                        BLE_CUSTOM_COMMAND_ENUM_INTEGER, //Type
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN, //MIN
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN,  //MAX
                        (void *) ValidIntValues, //Enum Array
                        NULL, //Enum String
                        "Example of Enum Integer", //Description
                        JSON_SensorArray)) {
      BLE_MANAGER_PRINTF("Added Command <%s>\r\n","FFT_Size");
    } else {
      BLE_MANAGER_PRINTF("Error Adding Command <%s>\r\n","FFT_Size");
      return;
    }
  }
  
  {
    //The Last value should be BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN
    int32_t ValidIntValues[]={8,16,32,64,BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN};
    if(AddCustomCommand("NumberOfSubrange", //Name
                        BLE_CUSTOM_COMMAND_ENUM_INTEGER, //Type
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN, //MIN
                        BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN,  //MAX
                        (void *) ValidIntValues, //Enum Array
                        NULL, //Enum String
                        "Example of Enum Integer", //Description
                        JSON_SensorArray)) {
      BLE_MANAGER_PRINTF("Added Command <%s>\r\n","NumberOfSubrange");
    } else {
      BLE_MANAGER_PRINTF("Error Adding Command <%s>\r\n","NumberOfSubrange");
      return;
    }
  }
  
  if(AddCustomCommand("FFT_Overlapping", //Name
                      BLE_CUSTOM_COMMAND_INTEGER, //Type
                      5, //MIN
                      95,  //MAX
                      NULL, //Enum Array
                      NULL, //Enum String
                      NULL, //Description
                      JSON_SensorArray)) {
    BLE_MANAGER_PRINTF("Added Command <%s>\r\n","FFT_Overlapping");
  } else {
     BLE_MANAGER_PRINTF("Error Adding Command <%s>\r\n","FFT_Overlapping");
     return;
  }
  
  if(AddCustomCommand("AquisitionTime", //Name
                      BLE_CUSTOM_COMMAND_INTEGER, //Type
                      1000, //MIN
                      10000,  //MAX
                      NULL, //Enum Array
                      NULL, //Enum String
                      NULL, //Description
                      JSON_SensorArray)) {
    BLE_MANAGER_PRINTF("Added Command <%s>\r\n","AquisitionTime");
  } else {
     BLE_MANAGER_PRINTF("Error Adding Command <%s>\r\n","AquisitionTime");
     return;
  }
}

/**
 * @brief  Callback Function for answering to the UID command
 * @param  uint8_t **UID STM32 UID Return value
 * @retval None
 */
static void ExtExtConfigUidCommandCallback(uint8_t **UID)
{
  *UID = (uint8_t *)STM32_UUID;
}


/**
 * @brief  Callback Function for answering to Info command
 * @param  uint8_t *Answer Return String
 * @retval None
 */
static void ExtConfigInfoCommandCallback(uint8_t *Answer)
{
  uint8_t  hwVersion;
  uint16_t fwVersion;
  
  /* get the BlueNRG HW and FW versions */
  getBlueNRG2_Version(&hwVersion, &fwVersion);

  sprintf((char *)Answer,"STMicroelectronics %s:\n"
    "Version %c.%c.%c\n"
    "%s board\n"
    "BlueNRG2 HW ver%d.%d\n"
    "BlueNRG2 FW ver%d.%d.%c\n"
    "(HAL %ld.%ld.%ld_%ld)\n"
    "Compiled %s %s"
#if defined (__IAR_SYSTEMS_ICC__)
    " (IAR)",
#elif defined (__CC_ARM)
    " (KEIL)",
#elif defined (__GNUC__)
    " (STM32CubeIDE)",
#endif
    BLE_FW_PACKAGENAME,
    BLE_VERSION_FW_MAJOR,
    BLE_VERSION_FW_MINOR,
    BLE_VERSION_FW_PATCH,
    BLE_STM32_BOARD,
    ((hwVersion>>4)&0x0F),
    (hwVersion&0x0F),
    (fwVersion>>8)&0xF,
    (fwVersion>>4)&0xF,
    ('a' + (fwVersion&0xF)),
    HAL_GetHalVersion() >>24,
    (HAL_GetHalVersion() >>16)&0xFF,
    (HAL_GetHalVersion() >> 8)&0xFF,
    HAL_GetHalVersion()      &0xFF,
    __DATE__,__TIME__);
}

/**
 * @brief  Callback Function for answering to Help command
 * @param  uint8_t *Answer Return String
 * @retval None
 */
static void ExtConfigHelpCommandCallback(uint8_t *Answer)
{
  sprintf((char *)Answer,"List of available command:\n"
                         "1) Board Report\n"
                         "- STM32 UID\n"
                         "- Version Firmware\n"
                         "- Info\n"
                         "- Help\n"
                         "- Power Status\n\n"
                         "2) Board Settings\n"
                         "- Set Name\n"
                         "- Read Custom Command\n");
}

/**
 * @brief  Callback Function for answering to PowerStatus command
 * @param  uint8_t *Answer Return String
 * @retval None
 */
static void ExtConfigPowerStatusCommandCallback(uint8_t *Answer)
{
  //sprintf((char *)Answer,"Plug the Battery");
  
  uint8_t Status[22];
  stbc02_State_TypeDef BC_State = {(stbc02_ChgState_TypeDef)0, ""};
  uint32_t BatteryLevel= 0;
  uint32_t Voltage;

  /* Read the voltage value and battery level status */
  BSP_BC_GetVoltageAndLevel(&Voltage,&BatteryLevel);
  
  BSP_BC_GetState(&BC_State);
  
  switch(BC_State.Id) {
    case VbatLow:
      sprintf((char *)Status,"Low Battery");
    break;
    case NotValidInput:
      sprintf((char *)Status,"Discharging");
    break;
    case EndOfCharge:
      sprintf((char *)Status,"Plugged Not Charging");
    break;
    case ChargingPhase:
      sprintf((char *)Status,"Charging");
    break;
    default:
      /* All the Remaing Battery Status */
      sprintf((char *)Status,"Unknown");
  }
  
  sprintf((char *)Answer,"Battery Status:\n"
                         "- BatteryLevel = %ld \n"
                         "- Voltage = %ld mv\n"
                         "- Status = %s\n",
                           BatteryLevel,
                           Voltage,
                           Status);
}
  
/**
 * @brief  Callback Function for answering to VersionFw command
 * @param  uint8_t *Answer Return String
 * @retval None
 */
static void ExtConfigVersionFwCommandCallback(uint8_t *Answer)
{
  sprintf((char *)Answer,"%s_%s_%c.%c.%c",
      BLE_STM32_MICRO,
      BLE_FW_PACKAGENAME,
      BLE_VERSION_FW_MAJOR,
      BLE_VERSION_FW_MINOR,
      BLE_VERSION_FW_PATCH);
}

/**
 * @brief  Callback Function for managing the SetName command
 * @param  uint8_t *NewName
 * @retval None
 */
static void ExtConfigSetNameCommandCallback(uint8_t *NewName)
{ 
  BLE_MANAGER_PRINTF("New Board Name = <%s>\r\n", NewName);
  /* Change the Board Name */
  sprintf(BlueNRG_StackValue.BoardName,"%s",NewName);
  
  for(int i=0; i<7; i++)
    NodeName[i+1]= BlueNRG_StackValue.BoardName[i];
  
  MDM_SaveGMD(GMD_NODE_NAME,(void *)&NodeName);
  NecessityToSaveMetaDataManager=1;
  
  BLE_MANAGER_PRINTF("\nThe node nome has been updated\r\n");
  BLE_MANAGER_PRINTF("Disconnecting and riconnecting to see the new node name\r\n");
}

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
