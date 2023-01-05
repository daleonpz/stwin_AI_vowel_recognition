/**
 ******************************************************************************
 * @file    main.c
 * @author  System Research & Applications Team - Catania Lab.
 * @version V2.4.0
 * @date    07-June-2021
 * @brief   Main program body
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

#include "main.h"
#include "hci.h"

/* Private define ------------------------------------------------------------*/
#define CHECK_VIBRATION_PARAM ((uint16_t)0x1234)

/* Imported Variables -------------------------------------------------------------*/

//#ifdef PREDMNT1_ENABLE_PRINTF
//extern TIM_HandleTypeDef  TimHandle;
//extern void CDC_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim);
//#endif /* PREDMNT1_ENABLE_PRINTF */

/* Exported Variables -------------------------------------------------------------*/
volatile uint32_t HCI_ProcessEvent=      0;
volatile uint8_t FifoEnabled = 0;

volatile uint32_t PredictiveMaintenance = 0;

float RMS_Ch[AUDIO_IN_CHANNELS];
float DBNOISE_Value_Old_Ch[AUDIO_IN_CHANNELS];

uint32_t ConnectionBleStatus  =0;

TIM_HandleTypeDef    TimCCHandle;

uint8_t EnvironmentalTimerEnabled= 0;
uint8_t AudioLevelTimerEnabled= 0;
uint8_t InertialTimerEnabled= 0;
uint8_t BatteryTimerEnabled= 0;

uint8_t AudioLevelEnable= 0;

uint32_t uhCCR1_Val = DEFAULT_uhCCR1_Val;
uint32_t uhCCR2_Val = DEFAULT_uhCCR2_Val;
uint32_t uhCCR3_Val = DEFAULT_uhCCR3_Val;
uint32_t uhCCR4_Val = DEFAULT_uhCCR4_Val;

uint8_t  NodeName[8];

/* Private variables ---------------------------------------------------------*/
uint16_t VibrationParam[11];

/* Table with All the known Meta Data */
MDM_knownGMD_t known_MetaData[]={
    {GMD_NODE_NAME,      (sizeof(NodeName))},
    {GMD_VIBRATION_PARAM,(sizeof(VibrationParam))},
    {GMD_END    ,0}/* THIS MUST BE THE LAST ONE */
};

static volatile uint32_t ButtonPressed=         0;
static volatile uint32_t SendEnv=               0;
static volatile uint32_t SendAudioLevel=        0;
static volatile uint32_t SendAccGyroMag=        0;
static volatile uint32_t SendBatteryInfo=       0;
static volatile uint32_t t_stwin=               0;

static volatile uint8_t  g_led_on           = 0;
uint32_t NumSample= ((AUDIO_IN_CHANNELS*AUDIO_IN_SAMPLING_FREQUENCY)/1000)  * N_MS;

/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);

static void InitTimers(void);
static void InitPredictiveMaintenance(void);

static unsigned char ReCallNodeNameFromMemory(void);
static unsigned char ReCallVibrationParamFromMemory(void);

static void SendEnvironmentalData(void);
static void SendMotionData(void);
static void SendAudioLevelData(void);
static void SendBatteryInfoData(void);

static void AudioProcess(void);
static void AudioProcess_DB_Noise(void);

static void Environmental_StartStopTimer(void);
static void AudioLevel_StartStopTimer(void);
static void Inertial_StartStopTimer(void);
static void BatteryFeatures_StartStopTimer(void);

static void FFTAmplitude_EnableDisableFeature(void);
static void FFTAlarmSpeedRMSStatus_EnableDisableFeature(void);
static void FFTAlarmAccPeakStatus_EnableDisableFeature(void);
static void FFTAlarmSubrangeStatus_EnableDisableFeature(void);


/**
 * @brief  Main program
 * @param  None
 * @retval None
 */
int main(void)
{
    HAL_Init();

    HAL_PWREx_EnableVddIO2();
    __HAL_RCC_PWR_CLK_ENABLE();
    HAL_PWREx_EnableVddUSB();

    /* Configure the System clock */
    SystemClock_Config();

    InitTargetPlatform();

    t_stwin = HAL_GetTick();

    /* Check the MetaDataManager */
    InitMetaDataManager((void *)&known_MetaData,MDM_DATA_TYPE_GMD,NULL); 

    PREDMNT1_PRINTF("\n\t(HAL %ld.%ld.%ld_%ld)\r\n"
            "\tCompiled %s %s"

#if defined (__IAR_SYSTEMS_ICC__)
            " (IAR)\r\n"
#elif defined (__CC_ARM)
            " (KEIL)\r\n"
#elif defined (__GNUC__)
            " (STM32CubeIDE)\r\n"
#endif
            "\tSend Every %4dmS Temperature/Humidity/Pressure\r\n"
            "\tSend Every %4dmS Acc/Gyro/Magneto\r\n"
            "\tSend Every %4dmS dB noise\r\n\n",
            HAL_GetHalVersion() >>24,
            (HAL_GetHalVersion() >>16)&0xFF,
            (HAL_GetHalVersion() >> 8)&0xFF,
            HAL_GetHalVersion()      &0xFF,
            __DATE__,__TIME__,
            ALGO_PERIOD_ENV,
            ALGO_PERIOD_ACC_GYRO_MAG,
            ALGO_PERIOD_AUDIO_LEVEL);

#ifdef PREDMNT1_DEBUG_CONNECTION
    PREDMNT1_PRINTF("Debug Connection         Enabled\r\n");
#endif /* PREDMNT1_DEBUG_CONNECTION */

#ifdef PREDMNT1_DEBUG_NOTIFY_TRAMISSION
    PREDMNT1_PRINTF("Debug Notify Trasmission Enabled\r\n\n");
#endif /* PREDMNT1_DEBUG_NOTIFY_TRAMISSION */

    HCI_TL_SPI_Reset();

    /* Check the BootLoader Compliance */
    PREDMNT1_PRINTF("\r\n");
    if(CheckBootLoaderCompliance()) {
        PREDMNT1_PRINTF("BootLoader Compliant with FOTA procedure\r\n\n");
    } else {
        PREDMNT1_PRINTF("ERROR: BootLoader NOT Compliant with FOTA procedure\r\n\n");
    }

    /* initialize timers */
    InitTimers();

    /* Predictive Maintenance Initialization */
    InitPredictiveMaintenance();

//     Environmental_StartStopTimer();
    Inertial_StartStopTimer();   


    uint32_t num_samples_1_sek = 1; //FREQ_ACC_GYRO_MAG;
    uint32_t samples_count = 0;
    /* Infinite loop */
    while (1)
    {

        if ( samples_count == 0 ) {
//             _PRINTF("[");
            LedOnTargetPlatform();
            samples_count++;
        }
        else if ( samples_count > num_samples_1_sek)
        {
//             _PRINTF("]\r\n");
            _PRINTF("\r\n");
            LedOffTargetPlatform();
//             HAL_Delay(1000); // 2sek
            samples_count = 0;
        }
        else {
//         if ( !g_led_on ) {
//            LedOnTargetPlatform();
//         }
//         else{
//             LedOffTargetPlatform();
//         }
//         Environmental_StartStopTimer();

//         /* Audio Level Features */
//             AudioLevel_StartStopTimer(); 
// 
//         /* Inertial Features */
//             Inertial_StartStopTimer();   
// 
//         /* Battery Features */
//             BatteryFeatures_StartStopTimer();
// 
//         /* FFT Amplitude Features */
//             FFTAmplitude_EnableDisableFeature();
// 
//         /* FFT FFT Alarm Speed Status Features */
//             FFTAlarmSpeedRMSStatus_EnableDisableFeature();      
// 
//         /* FFT Alarm Acc Peak Status Features */
//             FFTAlarmAccPeakStatus_EnableDisableFeature();
// 
//         /* FFT Alarm Subrange Status Features */
//             FFTAlarmSubrangeStatus_EnableDisableFeature(); 

// 
//         if(PredictiveMaintenance){
//             /* Manage the vibration analysis */
//             if (MotionSP_MainManager() != BSP_ERROR_NONE)
//                 Error_Handler();
//         }
// 
//         /* handle BLE event */
//         if(HCI_ProcessEvent) {
//             HCI_ProcessEvent=0;
//             hci_user_evt_proc();
//         }
//

        /* Environmental Data */
//         if(SendEnv) {
//             SendEnv=0;
//             SendEnvironmentalData();
//         }

//         /* Mic Data */
//         if (SendAudioLevel) {
//             SendAudioLevel = 0;
//             SendAudioLevelData();
//         }
// 
//         /* Motion Data */
        if(SendAccGyroMag) {
            SendAccGyroMag=0;
            SendMotionData();
            samples_count++;

        }
//         /* Battery Info Data */
//         if(SendBatteryInfo){
//             SendBatteryInfo=0;
//             SendBatteryInfoData();
//         }
// 
        /* Wait for Event */
//         __WFI();
        }
    }
}

/**
 * @brief  Send Motion Data Acc/Mag/Gyro to BLE
 * @param  None
 * @retval None
 */
static void SendMotionData(void)
{
    MOTION_SENSOR_Axes_t ACC_Value;
    MOTION_SENSOR_Axes_t GYR_Value;
    MOTION_SENSOR_Axes_t MAG_Value;

    /* Reset the Acc values */
    ACC_Value.x = ACC_Value.y = ACC_Value.z =0;

    /* Reset the Gyro values */
    GYR_Value.x = GYR_Value.y = GYR_Value.z =0;

    /* Reset the Magneto values */
    MAG_Value.x = MAG_Value.y = MAG_Value.z =0;

    TargetBoardFeatures.AccSensorIsInit = 1;
    TargetBoardFeatures.GyroSensorIsInit= 0;
    TargetBoardFeatures.MagSensorIsInit = 0;

//     _PRINTF("[");

    /* Read the Acc values */
    if(TargetBoardFeatures.AccSensorIsInit)
    {
        MOTION_SENSOR_GetAxes(ACCELERO_INSTANCE, MOTION_ACCELERO, &ACC_Value);
        _PRINTF("%d, %d, %d ", ACC_Value.x, ACC_Value.y, ACC_Value.z);;
    }

    /* Read the Gyro values */
    if(TargetBoardFeatures.GyroSensorIsInit)
    {
        MOTION_SENSOR_GetAxes(GYRO_INSTANCE,MOTION_GYRO, &GYR_Value);
        _PRINTF("%d, %d, %d ", GYR_Value.x, GYR_Value.y, GYR_Value.z);;
//         PREDMNT1_PRINTF("Sending GYR: %d %d %d \r\n", GYR_Value.x/100, GYR_Value.y/100, GYR_Value.z/100); // from BLE_AccGyroMagUpdate
    }

    /* Read the Magneto values */
    if(TargetBoardFeatures.MagSensorIsInit)
    {
        MOTION_SENSOR_GetAxes(MAGNETO_INSTANCE, MOTION_MAGNETO, &MAG_Value);
        _PRINTF("%d, %d, %d ", MAG_Value.x, MAG_Value.y, MAG_Value.z);;

//         _PRINTF(", %d, %d, %d ", MAG_Value.x, MAG_Value.y, MAG_Value.z);;
    }
//     _PRINTF("],"); // this will be parse in python 
    // in python [[ x,y,z],] = [[x,y,z]] .. the last ',' will be ignored
}

/**
 * @brief  User function that is called when 1 ms of PDM data is available.
 * @param  none
 * @retval None
 */
static void AudioProcess(void)
{
    if(AudioLevelEnable)
    {
        AudioProcess_DB_Noise();
    }
}

/**
 * @brief  User function that is called when 1 ms of PDM data is available.
 * @param  none
 * @retval None
 */
static void AudioProcess_DB_Noise(void)
{

    if(AudioLevelEnable) {
        for(uint32_t i = 0; i < (NumSample/2); i++){
            for(uint32_t NumberMic=0;NumberMic<AUDIO_IN_CHANNELS;NumberMic++) {
                RMS_Ch[NumberMic] += (float)((int16_t)PCM_Buffer[i*AUDIO_IN_CHANNELS+NumberMic] * ((int16_t)PCM_Buffer[i*AUDIO_IN_CHANNELS+NumberMic]));
            }
        }
    }
}

/**
 * @brief  Send Audio Level Data (Ch1) to BLE
 * @param  None
 * @retval None
 */
// static void SendAudioLevelData(void)
// {
//     int32_t NumberMic;
//     uint16_t DBNOISE_Value_Ch[AUDIO_IN_CHANNELS];
// 
//     for(NumberMic=0;NumberMic<(AUDIO_IN_CHANNELS);NumberMic++) {
//         DBNOISE_Value_Ch[NumberMic] = 0;
// 
//         RMS_Ch[NumberMic] /= ((float)(NumSample/AUDIO_IN_CHANNELS)*ALGO_PERIOD_AUDIO_LEVEL);
// 
//         DBNOISE_Value_Ch[NumberMic] = (uint16_t)((120.0f - 20 * log10f(32768 * (1 + 0.25f * (AUDIO_VOLUME_INPUT /*AudioInVolume*/ - 4))) + 10.0f * log10f(RMS_Ch[NumberMic])) * 0.3f + DBNOISE_Value_Old_Ch[NumberMic] * 0.7f);
//         DBNOISE_Value_Old_Ch[NumberMic] = DBNOISE_Value_Ch[NumberMic];
//         RMS_Ch[NumberMic] = 0.0f;
//     }
// 
//     BLE_AudioLevelUpdate(DBNOISE_Value_Ch, AUDIO_IN_CHANNELS);
// }

/**
 * @brief  Read Environmental Data (Temperature/Pressure/Humidity) from sensor
 * @param  int32_t *PressToSend
 * @param  uint16_t *HumToSend
 * @param  int16_t *Temp1ToSend
 * @param  int16_t *Temp2ToSend
 * @retval None
 */
void ReadEnvironmentalData(int32_t *PressToSend,uint16_t *HumToSend,int16_t *Temp1ToSend,int16_t *Temp2ToSend)
{
    float SensorValue;
    int32_t decPart, intPart;

    *PressToSend=0;
    *HumToSend=0;
    *Temp2ToSend=0,*Temp1ToSend=0;

    /* Read Humidity */
    if(TargetBoardFeatures.HumSensorIsInit) {
        ENV_SENSOR_GetValue(HUMIDITY_INSTANCE,ENV_HUMIDITY,&SensorValue);
        MCR_BLUEMS_F2I_1D(SensorValue, intPart, decPart);
        *HumToSend = intPart*10+decPart;
    }

    /* Read Temperature for sensor 1 */
    if(TargetBoardFeatures.TempSensorsIsInit[0]){
        ENV_SENSOR_GetValue(TEMPERATURE_INSTANCE_1,ENV_TEMPERATURE,&SensorValue);
        MCR_BLUEMS_F2I_1D(SensorValue, intPart, decPart);
        *Temp1ToSend = intPart*10+decPart;
    }

    /* Read Pressure */
    if(TargetBoardFeatures.PressSensorIsInit){
        ENV_SENSOR_GetValue(PRESSURE_INSTANCE,ENV_PRESSURE,&SensorValue);
        MCR_BLUEMS_F2I_2D(SensorValue, intPart, decPart);
        *PressToSend=intPart*100+decPart;
    }

    /* Read Temperature for sensor 2 */
    if(TargetBoardFeatures.TempSensorsIsInit[1]) {
        ENV_SENSOR_GetValue(TEMPERATURE_INSTANCE_2,ENV_TEMPERATURE,&SensorValue);
        MCR_BLUEMS_F2I_1D(SensorValue, intPart, decPart);
        *Temp2ToSend = intPart*10+decPart;
    }
}

/**
 * @brief  Send Environmetal Data (Temperature/Pressure/Humidity) to BLE
 * @param  None
 * @retval None
 */
static void SendEnvironmentalData(void)
{
    /* Pressure,Humidity, and Temperatures*/
    int32_t PressToSend;
    uint16_t HumToSend;
    int16_t Temp2ToSend,Temp1ToSend;

    /* Read all the Environmental Sensors */
    ReadEnvironmentalData(&PressToSend,&HumToSend, &Temp1ToSend,&Temp2ToSend);
    _PRINTF("Sending: Press=%ld Hum=%d Temp1=%d Temp2=%d \r\n", PressToSend, HumToSend, Temp1ToSend, Temp2ToSend);
}

/**
 * @brief  Send Battery Info Data (Voltage/Current/Soc) to BLE
 * @param  None
 * @retval None
 */
static void SendBatteryInfoData(void)
{
    uint32_t Status;  
    stbc02_State_TypeDef BC_State = {(stbc02_ChgState_TypeDef)0, ""};
    uint32_t BatteryLevel= 0;
    uint32_t Voltage;

    /* Read the voltage value and battery level status */
    BSP_BC_GetVoltageAndLevel(&Voltage,&BatteryLevel);

    BSP_BC_GetState(&BC_State);

    switch(BC_State.Id) {
        case VbatLow:
            Status = 0x00; /* Low Battery */
            break;
        case ValidInput:
            Status = 0x01; /* Discharging */
            break;
        case EndOfCharge:
            Status = 0x02; /* End of Charging == Plugged not Charging */
            break;
        case ChargingPhase:
            Status = 0x03; /* Charging */
            break;
        default:
            /* All the Remaing Battery Status */
            Status = 0x04; /* Unknown */
    }

    BLE_BatteryUpdate(BatteryLevel, Voltage, 0x8000, Status);

#ifdef PREDMNT1_DEBUG_NOTIFY_TRAMISSION
    /* Battery Informations */
    if(BLE_StdTerm_Service==BLE_SERV_ENABLE) {
        BytesToWrite = sprintf((char *)BufferToWrite,"Battery Report: \r\n");
        Term_Update(BufferToWrite,BytesToWrite);
        BytesToWrite = sprintf((char *)BufferToWrite,"Charge= %ld%% Voltage=%ld mV BC_State= %d\r\n", (long)BatteryLevel, (long)Voltage, BC_State.Id);
        Term_Update(BufferToWrite,BytesToWrite);
    } else {
        PREDMNT1_PRINTF("Battery Report: ");
        PREDMNT1_PRINTF("Charge= %ld%% Voltage=%ld mV BC_State= %d\r\n", BatteryLevel, Voltage, BC_State.Id);
    }
#endif /* PREDMNT1_DEBUG_NOTIFY_TRAMISSION */
}

/**
 * @brief  Function for initializing timers for sending the information to BLE:
 *  - 1 for sending MotionFX/AR/CP and Acc/Gyro/Mag
 *  - 1 for sending the Environmental info
 * @param  None
 * @retval None
 */
static void InitTimers(void)
{
    uint32_t uwPrescalerValue;

    /* Timer Output Compare Configuration Structure declaration */
    TIM_OC_InitTypeDef sConfig;

    /* Compute the prescaler value to have TIM1 counter clock equal to 10 KHz */
    uwPrescalerValue = (uint32_t) ((SystemCoreClock / 10000) - 1); 

    PREDMNT1_PRINTF("system clock ----> %d\r\n", SystemCoreClock);

    /* Set TIM1 instance ( Motion ) */
    TimCCHandle.Instance = TIM1;  
    TimCCHandle.Init.Period        = 65535;
    TimCCHandle.Init.Prescaler     = uwPrescalerValue;
    TimCCHandle.Init.ClockDivision = 0;
    TimCCHandle.Init.CounterMode   = TIM_COUNTERMODE_UP;
    if(HAL_TIM_OC_Init(&TimCCHandle) != HAL_OK)
    {
        /* Initialization Error */
        Error_Handler();
    }

    /* Configure the Output Compare channels */
    /* Common configuration for all channels */
    sConfig.OCMode     = TIM_OCMODE_TOGGLE;
    sConfig.OCPolarity = TIM_OCPOLARITY_LOW;

    /* Output Compare Toggle Mode configuration: Channel2 for environmental sensor */
    sConfig.Pulse = DEFAULT_uhCCR1_Val;
    if(HAL_TIM_OC_ConfigChannel(&TimCCHandle, &sConfig, TIM_CHANNEL_1) != HAL_OK)
    {
        /* Configuration Error */
        Error_Handler();
    }

    /* Output Compare Toggle Mode configuration: Channel2 for mic audio level */
    sConfig.Pulse = DEFAULT_uhCCR2_Val;
    if(HAL_TIM_OC_ConfigChannel(&TimCCHandle, &sConfig, TIM_CHANNEL_2) != HAL_OK)
    {
        /* Configuration Error */
        Error_Handler();
    }

    /* Output Compare Toggle Mode configuration: Channel3 for Acc/Gyro/Mag sensor */
    sConfig.Pulse = DEFAULT_uhCCR3_Val;
    if(HAL_TIM_OC_ConfigChannel(&TimCCHandle, &sConfig, TIM_CHANNEL_3) != HAL_OK)
    {
        /* Configuration Error */
        Error_Handler();
    }

    /* Output Compare Toggle Mode configuration: Channel4 for battery info */
    sConfig.Pulse = DEFAULT_uhCCR4_Val;
    if(HAL_TIM_OC_ConfigChannel(&TimCCHandle, &sConfig, TIM_CHANNEL_4) != HAL_OK)
    {
        /* Configuration Error */
        Error_Handler();
    }
}


/** @brief Predictive Maintenance Initialization
 * @param None
 * @retval None
 */
static void InitPredictiveMaintenance(void)
{
    /* Set the vibration parameters with default values */
    MotionSP_SetDefaultVibrationParam();

    /* Read Vibration Parameters From Memory */
    ReCallVibrationParamFromMemory();

    PREDMNT1_PRINTF("\r\nAccelerometer parameters:\r\n");
    PREDMNT1_PRINTF("AccOdr= %d\t", AcceleroParams.AccOdr);
    PREDMNT1_PRINTF("AccFifoBdr= %d\t", AcceleroParams.AccFifoBdr);   
    PREDMNT1_PRINTF("fs= %d\t", AcceleroParams.fs);   
    PREDMNT1_PRINTF("\r\n");

    PREDMNT1_PRINTF("\r\nMotionSP parameters:\r\n");
    PREDMNT1_PRINTF("size= %d\t", MotionSP_Parameters.FftSize); 
    PREDMNT1_PRINTF("wind= %d\t", MotionSP_Parameters.window);  
    PREDMNT1_PRINTF("tacq= %d\t", MotionSP_Parameters.tacq);
    PREDMNT1_PRINTF("ovl= %d\t", MotionSP_Parameters.FftOvl);
    PREDMNT1_PRINTF("subrange_num= %d\t", MotionSP_Parameters.subrange_num);
    PREDMNT1_PRINTF("\r\n\n");

    PREDMNT1_PRINTF("************************************************************************\r\n\r\n");

    /* Initializes accelerometer with vibration parameters values */
    if(MotionSP_AcceleroConfig()) {
        PREDMNT1_PRINTF("\tFailed Set Accelerometer Parameters\r\n\n");
    } else {
        PREDMNT1_PRINTF("\tOK Set Accelerometer Parameters\r\n\n");
    }
}

/**
 * @brief  System Clock Configuration
 *         The system Clock is configured as follow : 
 *            System Clock source            = PLL (HSI)
 *            SYSCLK(Hz)                     = 
 *            HCLK(Hz)                       = 
 *            AHB Prescaler                  = 1
 *            APB1 Prescaler                 = 1
 *            APB2 Prescaler                 = 1
 *            HSE Frequency(Hz)              = 
 *            PLL_M                          = 2
 *            PLL_N                          = 30
 *            PLL_P                          = 2
 *            PLL_Q                          = 2
 *            PLL_R                          = 2
 *            VDD(V)                         = 3.3
 *            Main regulator output voltage  = Scale1 mode
 *            Flash Latency(WS)              = 5
 * @param  None
 * @retval None
 */
void SystemClock_Config(void)
{
    //   __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    //  
    //  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    //  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    //  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    RCC_OscInitTypeDef RCC_OscInitStruct;
    RCC_ClkInitTypeDef RCC_ClkInitStruct;
    RCC_PeriphCLKInitTypeDef PeriphClkInit;

    /* Configure the main internal regulator output voltage */
    if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST) != HAL_OK)
    {
        Error_Handler();
    }

    /* Initializes the CPU, AHB and APB busses clocks */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48|RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;  // External crystal   (32khz/16khz)
    RCC_OscInitStruct.HSI48State = RCC_HSI48_ON; //48 Mhz Source to drive usbe
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 2;
    RCC_OscInitStruct.PLL.PLLN = 30;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV5; //RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
    RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }

    /* Initializes the CPU, AHB and APB busses clocks */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
        |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
    {
        Error_Handler();
    }

    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_RTC
        |RCC_PERIPHCLK_DFSDM1
        |RCC_PERIPHCLK_USB
        |RCC_PERIPHCLK_ADC
        |RCC_PERIPHCLK_I2C2
        |RCC_PERIPHCLK_SAI1;
    PeriphClkInit.Sai1ClockSelection = RCC_SAI1CLKSOURCE_PLLSAI1;
    PeriphClkInit.I2c2ClockSelection = RCC_I2C2CLKSOURCE_PCLK1;
    PeriphClkInit.AdcClockSelection = RCC_ADCCLKSOURCE_PLLSAI1;
    PeriphClkInit.Dfsdm1ClockSelection = RCC_DFSDM1CLKSOURCE_PCLK2; //RCC_DFSDM1CLKSOURCE_PCLK;
    PeriphClkInit.RTCClockSelection = RCC_RTCCLKSOURCE_LSE;
    PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_HSI48;
    PeriphClkInit.PLLSAI1.PLLSAI1Source = RCC_PLLSOURCE_HSE;
    PeriphClkInit.PLLSAI1.PLLSAI1M = 5;
    PeriphClkInit.PLLSAI1.PLLSAI1N = 96;
    PeriphClkInit.PLLSAI1.PLLSAI1P = RCC_PLLP_DIV25;
    PeriphClkInit.PLLSAI1.PLLSAI1Q = RCC_PLLQ_DIV4;
    PeriphClkInit.PLLSAI1.PLLSAI1R = RCC_PLLR_DIV4;
    PeriphClkInit.PLLSAI1.PLLSAI1ClockOut = RCC_PLLSAI1_ADC1CLK|RCC_PLLSAI1_SAI1CLK;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
    {
        Error_Handler();
    }

    /* Configure the Systick interrupt time */
    HAL_SYSTICK_Config(HAL_RCC_GetHCLKFreq()/1000);

    /* Configure the Systick */
    HAL_SYSTICK_CLKSourceConfig(SYSTICK_CLKSOURCE_HCLK);

    /* SysTick_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(SysTick_IRQn, 0, 0);
}

/**
 * @brief  Check if there are a valid Vibration Parameters Values in Memory and read them
 * @param pAccelerometer_Parameters Pointer to Accelerometer parameter structure
 * @param pMotionSP_Parameters Pointer to Board parameter structure
 * @retval unsigned char Success/Not Success
 */
static unsigned char ReCallVibrationParamFromMemory(void)
{
    /* ReLoad the Vibration Parameters Values from RAM */
    unsigned char Success=0;

    PREDMNT1_PRINTF("Recall the vibration parameter values from FLASH\r\n");

    /* Recall the Vibration Parameters Values saved */
    MDM_ReCallGMD(GMD_VIBRATION_PARAM,(void *)VibrationParam);

    if(VibrationParam[0] == CHECK_VIBRATION_PARAM)
    {
        AcceleroParams.AccOdr=              VibrationParam[1];
        AcceleroParams.AccFifoBdr=          VibrationParam[2];
        AcceleroParams.fs=                  VibrationParam[3];
        MotionSP_Parameters.FftSize=        VibrationParam[4];
        MotionSP_Parameters.tau=            VibrationParam[5];
        MotionSP_Parameters.window=         VibrationParam[6];
        MotionSP_Parameters.td_type=        VibrationParam[7];
        MotionSP_Parameters.tacq=           VibrationParam[8];
        MotionSP_Parameters.FftOvl=         VibrationParam[9];
        MotionSP_Parameters.subrange_num=   VibrationParam[10];

        PREDMNT1_PRINTF("Vibration parameter values read from FLASH\r\n");

        NecessityToSaveMetaDataManager=0;
    }
    else
    {
        PREDMNT1_PRINTF("Vibration parameters values not present in FLASH\r\n");
        SaveVibrationParamToMemory();
    }

    return Success;
}


/*******************************************/
/* Hardware Characteristics Notify Service */
/*******************************************/

/**
 * @brief  This function is called when there is a change on the gatt attribute for Environmental
 *         for Start/Stop Timer
 * @param  None
 * @retval None
 */
static void Environmental_StartStopTimer(void)
{
    if(!EnvironmentalTimerEnabled) {
        /* Start the TIM Base generation in interrupt mode (for environmental sensor) */
        if(HAL_TIM_OC_Start_IT(&TimCCHandle, TIM_CHANNEL_1) != HAL_OK){
            /* Starting Error */
            Error_Handler();
        }

        /* Set the new Capture compare value */
        {
            uint32_t uhCapture = __HAL_TIM_GET_COUNTER(&TimCCHandle);
            /* Set the Capture Compare Register value (for environmental) */
            __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_1, (uhCapture + uhCCR1_Val));
        }

        EnvironmentalTimerEnabled = 1;
    }

}

/**
 * @brief  This function is called when there is a change on the gatt attribute for Audio Level
 *         for Start/Stop Timer
 * @param  None
 * @retval None
 */
// static void AudioLevel_StartStopTimer(void)
// {
//     if( (BLE_AudioLevel_NotifyEvent == BLE_NOTIFY_SUB) &&
//             (!AudioLevelTimerEnabled) ) {
//         int32_t Count;
// 
//         InitMics(AUDIO_IN_SAMPLING_FREQUENCY, AUDIO_VOLUME_INPUT);
//         AudioLevelEnable= 1;
// 
//         for(Count=0;Count<AUDIO_IN_CHANNELS;Count++) {
//             RMS_Ch[Count]=0;
//             DBNOISE_Value_Old_Ch[Count] =0;
//         }
// 
//         /* Start the TIM Base generation in interrupt mode (for mic audio level) */
//         if(HAL_TIM_OC_Start_IT(&TimCCHandle, TIM_CHANNEL_2) != HAL_OK){
//             /* Starting Error */
//             Error_Handler();
//         }
// 
//         /* Set the new Capture compare value */
//         {
//             uint32_t uhCapture = __HAL_TIM_GET_COUNTER(&TimCCHandle);
//             /* Set the Capture Compare Register value (for mic audio level) */
//             __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_2, (uhCapture + uhCCR2_Val));
//         }
// 
//         AudioLevelTimerEnabled= 1;
//     }
// 
//     if( (BLE_AudioLevel_NotifyEvent == BLE_NOTIFY_UNSUB) &&
//             (AudioLevelTimerEnabled) ) {
//         DeInitMics();
//         AudioLevelEnable= 0;
// 
//         /* Stop the TIM Base generation in interrupt mode (for mic audio level) */
//         if(HAL_TIM_OC_Stop_IT(&TimCCHandle, TIM_CHANNEL_2) != HAL_OK){
//             /* Stopping Error */
//             Error_Handler();
//         }  
// 
//         AudioLevelTimerEnabled= 0;
//     }
// }

/**
 * @brief  This function is called when there is a change on the gatt attribute for inertial
 *         for Start/Stop Timer
 * @param  None
 * @retval None
 */
static void Inertial_StartStopTimer(void)
{ 
    if( !InertialTimerEnabled ){
    /* Start the TIM Base generation in interrupt mode (for Acc/Gyro/Mag sensor) */
        if(HAL_TIM_OC_Start_IT(&TimCCHandle, TIM_CHANNEL_3) != HAL_OK){
            /* Starting Error */
            Error_Handler();
        }

        /* Set the new Capture compare value */
        {
            uint32_t uhCapture = __HAL_TIM_GET_COUNTER(&TimCCHandle);
            /* Set the Capture Compare Register value (for Acc/Gyro/Mag sensor) */
            __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_3, (uhCapture + uhCCR3_Val));
        }

        InertialTimerEnabled= 1;
    }

}

/**
 * @brief  This function is called when there is a change on the gatt attribute for Battery Features
 *         for Start/Stop Timer
 * @param  None
 * @retval None
 */
// static void BatteryFeatures_StartStopTimer(void)
// {
//     if( (BLE_Battery_NotifyEvent == BLE_NOTIFY_SUB) && (!BatteryTimerEnabled) ){
// 
//         BSP_BC_CmdSend(BATMS_ON);
// 
//         /* Start the TIM Base generation in interrupt mode (for battery info) */
//         if(HAL_TIM_OC_Start_IT(&TimCCHandle, TIM_CHANNEL_4) != HAL_OK){
//             /* Starting Error */
//             Error_Handler();
//         }
// 
//         /* Set the new Capture compare value */
//         {
//             uint32_t uhCapture = __HAL_TIM_GET_COUNTER(&TimCCHandle);
//             /* Set the Capture Compare Register value (for Acc/Gyro/Mag sensor) */
//             __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_4, (uhCapture + uhCCR4_Val));
//         }
// 
//         BatteryTimerEnabled= 1;
//     }
// 
//     if( (BLE_Battery_NotifyEvent == BLE_NOTIFY_UNSUB) && (BatteryTimerEnabled) ) {
//         /* Stop the TIM Base generation in interrupt mode (for battery info) */
//         if(HAL_TIM_OC_Stop_IT(&TimCCHandle, TIM_CHANNEL_4) != HAL_OK){
//             /* Stopping Error */
//             Error_Handler();
//         }
// 
//         BatteryTimerEnabled= 0;
// 
//         BSP_BC_CmdSend(BATMS_OFF);
//     }
// }

/***********************************
 * Software Characteristics Service *
 ************************************/

/**
 * @brief  This function is called when there is a change on the gatt attribute for FFT Amplitude
 *         for Enable/Disable Feature
 * @param  None
 * @retval None
 */
// static void FFTAmplitude_EnableDisableFeature(void)
// {
//     if(BLE_FFT_Amplitude_NotifyEvent == BLE_NOTIFY_SUB) {
//         PredictiveMaintenance= 1;
//         FFT_Amplitude= 1;
//     }
// 
//     if(BLE_FFT_Amplitude_NotifyEvent == BLE_NOTIFY_UNSUB) {
//         disable_FIFO();
//         EnableDisable_ACC_HP_Filter(0);
//         PredictiveMaintenance= 0;
//         FFT_Amplitude= 0;
//         MotionSP_Running = 0;
//     }
// }

/**
 * @brief  This function is called when there is a change on the gatt attribute for FFT Alarm Speed RMS
 *         for Enable/Disable Feature
 * @param  None
 * @retval None
 */
// static void FFTAlarmSpeedRMSStatus_EnableDisableFeature(void)
// {
//     if(BLE_FFTAlarmSpeedStatus_NotifyEvent == BLE_NOTIFY_SUB) {
//         if(!PredictiveMaintenance)
//         {
//             PredictiveMaintenance= 1;
//             FFT_Alarm= 1;
//         }
//     }
// 
//     if(BLE_FFTAlarmSpeedStatus_NotifyEvent == BLE_NOTIFY_UNSUB) {
//         if(PredictiveMaintenance)
//         {
//             disable_FIFO();
//             EnableDisable_ACC_HP_Filter(0);
//             PredictiveMaintenance= 0;
//             FFT_Alarm= 0;
//             MotionSP_Running = 0;
//         }
//     }
// }

/**
 * @brief  This function is called when there is a change on the gatt attribute for FFT Alarm Acc Peak Status
 *         for Enable/Disable Feature 
 * @param  None
 * @retval None
 */
// static void FFTAlarmAccPeakStatus_EnableDisableFeature(void)
// {
//     if(BLE_FFTAlarmAccPeakStatus_NotifyEvent == BLE_NOTIFY_SUB) {
//         if(!PredictiveMaintenance)
//         {
//             PredictiveMaintenance= 1;
//             FFT_Alarm= 1;
//         }
//     }
// 
//     if(BLE_FFTAlarmAccPeakStatus_NotifyEvent == BLE_NOTIFY_UNSUB ){
//         if(PredictiveMaintenance)
//         {
//             disable_FIFO();
//             EnableDisable_ACC_HP_Filter(0);
//             PredictiveMaintenance= 0;
//             FFT_Alarm= 0;
//             MotionSP_Running = 0;
//         }
//     }
// }

/**
 * @brief  This function is called when there is a change on the gatt attribute for FFT Alarm Subrange Status
 *         for Enable/Disable Feature 
 * @param  None
 * @retval None
 */
// static void FFTAlarmSubrangeStatus_EnableDisableFeature(void)
// {
//     if(BLE_FFTAlarmSubrangeStatus_NotifyEvent == BLE_NOTIFY_SUB) {
//         if(!PredictiveMaintenance)
//         {
//             PredictiveMaintenance= 1;
//             FFT_Alarm= 1;
//         }
//     }
// 
//     if(BLE_FFTAlarmSubrangeStatus_NotifyEvent == BLE_NOTIFY_UNSUB) {
//         if(PredictiveMaintenance)
//         {
//             disable_FIFO();
//             EnableDisable_ACC_HP_Filter(0);
//             PredictiveMaintenance= 0;
//             FFT_Alarm= 0;
//             MotionSP_Running = 0;
//         }
//     }
// }

/**
 * @}
 */

/** @defgroup PREDCTIVE_MAINTENANCE_MAIN_CALLBACK_FUNCTIONS Predictive Maintenance Main CallBack Functions
 * @{
 */

/**
 * @brief  Output Compare callback in non blocking mode 
 * @param  htim : TIM OC handle
 * @retval None
 */
void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim)
{
    uint32_t uhCapture;

    g_led_on ^= 1; 
    /* TIM1_CH1 toggling with frequency = 2 Hz */
    if(htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1)
    {
        uhCapture = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);
        /* Set the Capture Compare Register value (for environmental sensor) */
        __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_1, (uhCapture + uhCCR1_Val));
        SendEnv=1;
    }

    /* TIM1_CH2 toggling with frequency = 20 Hz */
    if(htim->Channel == HAL_TIM_ACTIVE_CHANNEL_2)
    {
        uhCapture = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_2);
        /* Set the Capture Compare Register value (for mic audio level) */
        __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_2, (uhCapture + uhCCR2_Val));
        SendAudioLevel=1;
    }

    /* TIM1_CH3 toggling with frequency = 20 Hz */
    if(htim->Channel == HAL_TIM_ACTIVE_CHANNEL_3)
    {
        uhCapture = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_3);
        /* Set the Capture Compare Register value (for Acc/Gyro/Mag sensor) */
        __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_3, (uhCapture + uhCCR3_Val));
        SendAccGyroMag=1;
    }

    /* TIM1_CH4 toggling with frequency = 20 Hz */
    if(htim->Channel == HAL_TIM_ACTIVE_CHANNEL_4)
    {
        uhCapture = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_4);
        /* Set the Capture Compare Register value (for battery info) */
        __HAL_TIM_SET_COMPARE(&TimCCHandle, TIM_CHANNEL_4, (uhCapture + uhCCR4_Val));
        SendBatteryInfo=1;
    }
}

/**
 * @brief  Period elapsed callback in non blocking mode for Environmental timer
 * @param  htim : TIM handle
 * @retval None
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == STBC02_USED_TIM) {
        BC_CmdMng();
#ifdef PREDMNT1_ENABLE_PRINTF
    } else if(htim == (&TimHandle)) {
        CDC_TIM_PeriodElapsedCallback(htim);
#endif /* PREDMNT1_ENABLE_PRINTF */
    }
}

/**
 * @brief  Conversion complete callback in non blocking mode 
 * @param  htim : hadc handle
 * @retval None
 */
void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Channel == HAL_TIM_ACTIVE_CHANNEL_3)
    {
        BSP_BC_ChgPinHasToggled();
    }
}

/**
 * @brief  Half Transfer user callback, called by BSP functions.
 * @param  None
 * @retval None
 */
void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
    AudioProcess();
}

/**
 * @brief  Transfer Complete user callback, called by BSP functions.
 * @param  None
 * @retval None
 */
void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
    AudioProcess();
}

/**
 * @brief  EXTI line detection callback.
 * @param  uint16_t GPIO_Pin Specifies the pins connected EXTI line
 * @retval None
 */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{  
    switch(GPIO_Pin){
        //    case HCI_TL_SPI_EXTI_PIN:
        //      hci_tl_lowlevel_isr();
        //      HCI_ProcessEvent=1;
        //    break;
        case M_INT2_O_PIN:
            if(FifoEnabled)
                MotionSP_FifoFull_IRQ_Rtn();
            else
                MotionSP_DataReady_IRQ_Rtn();
            break;

        case GPIO_PIN_10:
            if(HAL_GetTick() - t_stwin > 4000)
            {
                BSP_BC_CmdSend(SHIPPING_MODE_ON);
            }
            break;
    }
}

/**
 * @brief This function provides accurate delay (in milliseconds) based 
 *        on variable incremented.
 * @note This is a user implementation using WFI state
 * @param Delay: specifies the delay time length, in milliseconds.
 * @retval None
 */
void HAL_Delay(__IO uint32_t Delay)
{
    uint32_t tickstart = 0;
    tickstart = HAL_GetTick();
    while((HAL_GetTick() - tickstart) < Delay){
        __WFI();
    }
}

/**
 * @brief  This function is executed in case of error occurrence.
 * @param  None
 * @retval None
 */
void Error_Handler(void)
{
    /* User may add here some code to deal with this error */
    while(1){
    }
}

/**
 * @brief  Save vibration parameters values to memory
 * @param pAccelerometer_Parameters Pointer to Accelerometer parameter structure
 * @param pMotionSP_Parameters Pointer to Board parameter structure
 * @retval unsigned char Success/Not Success
 */
unsigned char SaveVibrationParamToMemory(void)
{
    /* ReLoad the Vibration Parameters Values from RAM */
    unsigned char Success=0;

    VibrationParam[0]= CHECK_VIBRATION_PARAM;
    VibrationParam[1]=  (uint16_t)AcceleroParams.AccOdr;
    VibrationParam[2]=  (uint16_t)AcceleroParams.AccFifoBdr;
    VibrationParam[3]=  (uint16_t)AcceleroParams.fs;
    VibrationParam[4]=  (uint16_t)MotionSP_Parameters.FftSize;
    VibrationParam[5]=  (uint16_t)MotionSP_Parameters.tau;
    VibrationParam[6]=  (uint16_t)MotionSP_Parameters.window;
    VibrationParam[7]=  (uint16_t)MotionSP_Parameters.td_type;
    VibrationParam[8]=  (uint16_t)MotionSP_Parameters.tacq;
    VibrationParam[9]=  (uint16_t)MotionSP_Parameters.FftOvl;
    VibrationParam[10]= (uint16_t)MotionSP_Parameters.subrange_num;

    PREDMNT1_PRINTF("Vibration parameters values will be saved in FLASH\r\n");
    MDM_SaveGMD(GMD_VIBRATION_PARAM,(void *)VibrationParam);
    NecessityToSaveMetaDataManager=1;

    return Success;
}

/**
 * @}
 */

#ifdef  USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t* file, uint32_t line)
{ 
    /* User can add his own implementation to report the file name and line number,
ex: PREDMNT1_PRINTF("Wrong parameters value: file %s on line %d\r\n", file, line) */

    /* Infinite loop */
    while (1){
    }
}
#endif

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
