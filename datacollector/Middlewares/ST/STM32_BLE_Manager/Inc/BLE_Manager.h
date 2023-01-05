/**
  ******************************************************************************
  * @file    BLE_Manager.h 
  * @author  System Research & Applications Team - Agrate/Catania Lab.
  * @version V0.3.0
  * @date    18-Jan-2021
  * @brief   BLE Manager services APIs
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0094, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0094
  *
  ******************************************************************************
  */
  
/* Define to prevent recursive inclusion -------------------------------------*/  
#ifndef _BLE_MANAGER_H_
#define _BLE_MANAGER_H_

#ifdef __cplusplus
 extern "C" {
#endif 

/* Includes ------------------------------------------------------------------*/
#include <stdlib.h>

#include "parson.h"
#include "hci.h"
#include "bluenrg1_hal_aci.h"
#include "bluenrg1_gatt_aci.h"
#include "bluenrg1_gap_aci.h"
#include "bluenrg1_hci_le.h"
#include "bluenrg1_l2cap_aci.h"
   
#include "BLE_Manager_Conf.h"
   

/* Exported Defines ----------------------------------------------------------*/
 
/* BLE Manager Version only numbers 0->9 */
#define BLE_MANAGER_VERSION_MAJOR '0'
#define BLE_MANAGER_VERSION_MINOR '3'
#define BLE_MANAGER_VERSION_PATCH '0'
   
/* SDK value for used platform */
#define BLE_MANAGER_STEVAL_WESU1_PLATFORM       0x01U
#define BLE_MANAGER_SENSOR_TILE_PLATFORM        0x02U
#define BLE_MANAGER_BLUE_COIN_PLATFORM          0x03U
#define BLE_MANAGER_STEVAL_IDB008VX_PLATFORM    0x04U
#define BLE_MANAGER_STEVAL_BCN002V1_PLATFORM    0x05U
#define BLE_MANAGER_SENSOR_TILE_BOX_PLATFORM    0x06U
#define BLE_MANAGER_DISCOVERY_IOT01A_PLATFORM   0x07U
#define BLE_MANAGER_STEVAL_STWINKIT1_PLATFORM   0x08U
#define BLE_MANAGER_NUCLEO_PLATFORM             0x80U
#define BLE_MANAGER_UNDEF_PLATFORM              0xFFU
   
#define BLE_MANAGER_CUSTOM_COMMAND_VALUE_NAN 0xDEADBEEF
#define BLE_MANAGER_CUSTOM_COMMAND_MAX_LEGHT 32U
   
#ifdef BLE_MANAGER_SDKV2
#define BLE_MANAGER_CUSTOM_FIELD1 15
#define BLE_MANAGER_CUSTOM_FIELD2 16
#define BLE_MANAGER_CUSTOM_FIELD3 17
#define BLE_MANAGER_CUSTOM_FIELD4 18
#endif /* BLE_MANAGER_SDKV2 */
   
#define COM_TYPE_ACC    1
#define COM_TYPE_MAG    2
#define COM_TYPE_GYRO   3
#define COM_TYPE_TEMP   4
#define COM_TYPE_PRESS  5
#define COM_TYPE_HUM    6
#define COM_TYPE_MIC    7
#define COM_TYPE_MLC    8
   
#define DATA_TYPE_UINT8     (uint8_t)(0x00)
#define DATA_TYPE_INT8      (uint8_t)(0x01)
#define DATA_TYPE_UINT16    (uint8_t)(0x02)
#define DATA_TYPE_INT16     (uint8_t)(0x03)
#define DATA_TYPE_UINT32    (uint8_t)(0x04)
#define DATA_TYPE_INT32     (uint8_t)(0x05)
#define DATA_TYPE_FLOAT     (uint8_t)(0x06)
   
#define N_MAX_DIM_LABELS                    8U
#define DIM_LABELS_LENGTH                   3U
#define N_MAX_SENSOR_COMBO                  4U
#define N_MAX_SUPPORTED_ODR                 16U
#define N_MAX_SUPPORTED_FS                  16U
   
#define COM_END_OF_LIST_INT -1
#define COM_END_OF_LIST_FLOAT -1.0f

#define COM_LIST_SEPARATOR_INT -2
#define COM_LIST_SEPARATOR_FLOAT -2.0f
   
/* Exported Types ------------------------------------------------------------*/
   
typedef struct
{
  /* BlueNRG stack setting */
  uint8_t ConfigValueOffsets;
  uint8_t ConfigValuelength;
  uint8_t GAP_Roles;
  uint8_t IO_capabilities;
  uint8_t AuthenticationRequirements;
  uint8_t MITM_ProtectionRequirements;
  uint8_t SecureConnectionSupportOptionCode;
  uint8_t SecureConnectionKeypressNotification;
  
  /* To set the TX power level of the bluetooth device */
  /* ----------------------
     | 0x00: Normal Power |
     | 0x01: High Power   |
     ---------------------- */
  uint8_t EnableHighPowerMode;
  
  /* Values: 0x00 ... 0x31 - The value depends on the device */
  uint8_t PowerAmplifierOutputLevel;
  
  /* BLE Manager services setting */
  uint8_t EnableConfig;
  uint8_t EnableConsole;
  uint8_t EnableExtConfig;
  
  /* BLE Board Name */
  char BoardName[8];
  
  /* For enabling the Secure BLE connection */
  uint8_t EnableSecureConnection;
  
  /* Secure Connection PIN */
  uint32_t SecurePIN;
  
  /* For creating a Random Connection PIN */
  uint8_t EnableRandomSecurePIN;
  
  uint8_t AdvertisingFilter;
  
  /* Set to 1 for forcing a full BLE rescan for the Android/iOS "ST BLE Sensor" application */
  /* with the Secure connection it should not necessary because it will be managed directly by BLE Chip */
  uint8_t ForceRescan;
  
  /* Bluetooth Board Mac Address */
  uint8_t BleMacAddress[6];

} BlueNRG_StackTypeDef;

typedef struct
{
  // BLE Char Definition
  uint8_t uuid[16];
  uint8_t Char_UUID_Type;
  uint16_t Char_Value_Length;
  uint8_t Char_Properties;
  uint8_t Security_Permissions;
  uint8_t GATT_Evt_Mask;
  uint8_t Enc_Key_Size;
  uint8_t Is_Variable;

  //BLE Attribute handle
  uint16_t attr_handle;
  //BLE Service handle
  uint16_t Service_Handle;

  // Callback function pointers
  // Attribute Modify
  void (*AttrMod_Request_CB)(void *BleCharPointer,uint16_t attr_handle, uint16_t Offset, uint8_t data_length, uint8_t *att_data);
  // Read Request
  void (*Read_Request_CB)(void *BleCharPointer,uint16_t handle);
  // Write Request
  void (*Write_Request_CB)(void *BleCharPointer,uint16_t attr_handle, uint16_t Offset, uint8_t data_length, uint8_t *att_data);
} BleCharTypeDef;

//Enum type for Service Notification Change
typedef enum {
  BLE_NOTIFY_NOTHING = 0, //No Event
  BLE_NOTIFY_SUB     = 1, //Subscription Event
  BLE_NOTIFY_UNSUB   = 2  //Unsubscription Event
} BLE_NotifyEnv_t;

//Enum type for Standard Service Enabled or Not
typedef enum {
  BLE_SERV_NOT_ENABLE= 0, //Service Not Enable
  BLE_SERV_ENABLE    = 1  //Service Enabled
} BLE_ServEnab_t;

//Typedef for Wi-Fi credential
typedef struct {
  uint8_t *SSID;
  uint8_t *PassWd;
  uint8_t *Security;
} BLE_WiFi_CredAcc_t;

//Typedef for Custom Command types
typedef enum {
  BLE_CUSTOM_COMMAND_VOID,
  BLE_CUSTOM_COMMAND_INTEGER,
  BLE_CUSTOM_COMMAND_BOOLEAN,
  BLE_CUSTOM_COMMAND_STRING,
  BLE_CUSTOM_COMMAND_ENUM_INTEGER,
  BLE_CUSTOM_COMMAND_ENUM_STRING
} BLE_CustomCommandTypes_t;

//Typedef for Custom Commad
typedef struct
{
  uint8_t *CommandName; 
  BLE_CustomCommandTypes_t CommandType;
  int32_t IntValue;
  uint8_t *StringValue;
} BLE_CustomCommadResult_t;

typedef struct
{
  uint8_t    id;
  uint8_t    sensorType;
  uint8_t    dimensions;
  char       dimensionsLabel[N_MAX_DIM_LABELS][DIM_LABELS_LENGTH+1U];
  char       unit[16];
  uint8_t    dataType;
  float      FS[N_MAX_SUPPORTED_FS];
  float      ODR[N_MAX_SUPPORTED_ODR];
  uint16_t   samplesPerTimestamp[2];
}
COM_SubSensorDescriptor_t;

/* Context is only used in the firmware, it's not written into DeviceConfiG.json */
typedef struct
{
  float     n_samples_acc;          // sensor_n_samples_acc
  double    old_time_stamp;
  uint16_t  n_samples_to_timestamp; // sensor_n_samples_to_timestamp
  uint8_t   first_dataReady;        // sensor_first_dataReady
  uint8_t*  sd_write_buffer;
  uint32_t  sd_write_buffer_idx;
  //FIL       file_handler;
}
COM_SubSensorContext_t;

typedef struct
{
  uint8_t                 isActive;
  float                   ODR;
  float                   measuredODR;
  float                   initialOffset;
  uint16_t                samplesPerTimestamp;
  float                   FS;
  float                   sensitivity;
  uint16_t                usbDataPacketSize;
  uint32_t                sdWriteBufferSize;
  uint32_t                wifiDataPacketSize;
  int16_t                 comChannelNumber;
  uint8_t                 ucfLoaded;
  COM_SubSensorContext_t  context;
}
COM_SubSensorStatus_t;

typedef struct
{
  uint8_t                   id;
  char                      name[16];
  uint8_t                   nSubSensors;
  COM_SubSensorDescriptor_t subSensorDescriptor[N_MAX_SENSOR_COMBO];
} COM_SensorDescriptor_t;

typedef struct
{
  COM_SubSensorStatus_t  subSensorStatus[N_MAX_SENSOR_COMBO];
}
COM_SensorStatus_t;

typedef struct
{
  COM_SensorDescriptor_t sensorDescriptor;
  COM_SensorStatus_t     sensorStatus;
}
COM_Sensor_t;


/* Exported Variables ------------------------------------------------------- */

extern  BLE_ServEnab_t BLE_Conf_Service;
extern  BLE_ServEnab_t BLE_StdTerm_Service;
extern  BLE_ServEnab_t BLE_StdErr_Service;
extern  BLE_ServEnab_t BLE_ExtConf_Service;

extern uint8_t BufferToWrite[256];
extern uint8_t BytesToWrite;
extern uint8_t set_connectable;

extern BlueNRG_StackTypeDef BlueNRG_StackValue;

/* Exported Function prototypes --------------------------------------------- */

/**************** Bluetooth Comunication *************************/
typedef void (*CustomConnectionCompleted_t)(uint16_t ConnectionHandle);
extern CustomConnectionCompleted_t CustomConnectionCompleted;

typedef void (*CustomDisconnectionCompleted_t)(void);
extern CustomDisconnectionCompleted_t CustomDisconnectionCompleted;

typedef void (*CustomAciGattTxPoolAvailableEvent_t)(void);
extern CustomAciGattTxPoolAvailableEvent_t CustomAciGattTxPoolAvailableEvent;

/**************** Debug Console *************************/
typedef uint32_t (*CustomDebugConsoleParsing_t)(uint8_t * att_data, uint8_t data_length);
extern CustomDebugConsoleParsing_t CustomDebugConsoleParsingCallback;

/******************* Config Char *************************/
typedef void (*CustomWriteRequestConfig_t)(uint8_t * att_data, uint8_t data_length);
extern CustomWriteRequestConfig_t CustomWriteRequestConfigCallback;

/*********** Extended Configuration Char ****************/
//For Reboot on DFU Command
typedef void (*CustomExtConfigRebootOnDFUModeCommand_t)(void);
extern CustomExtConfigRebootOnDFUModeCommand_t CustomExtConfigRebootOnDFUModeCommandCallback;

//For Power off Command
typedef void (*CustomExtConfigPowerOffCommand_t)(void);
extern CustomExtConfigPowerOffCommand_t CustomExtConfigPowerOffCommandCallback;

//For Set board Name Command
typedef void (*CustomExtConfigSetNameCommand_t)(uint8_t *NewName);
extern CustomExtConfigSetNameCommand_t CustomExtConfigSetNameCommandCallback;

//For Set Time Command
typedef void (*CustomExtConfigSetTimeCommand_t)(uint8_t *NewTime);
extern CustomExtConfigSetTimeCommand_t CustomExtConfigSetTimeCommandCallback;

//For Set Date Command
typedef void (*CustomExtConfigSetDateCommand_t)(uint8_t *NewDate);
extern CustomExtConfigSetDateCommand_t CustomExtConfigSetDateCommandCallback;

//For Set Wi-Fi Command
typedef void (*CustomExtConfigSetWiFiCommand_t)(BLE_WiFi_CredAcc_t NewWiFiCred);
extern CustomExtConfigSetWiFiCommand_t CustomExtConfigSetWiFiCommandCallback;

//For Change Secure PIN Command
typedef void (*CustomExtConfigChangePinCommand_t)(uint32_t NewPin);
extern CustomExtConfigChangePinCommand_t CustomExtConfigChangePinCommandCallback;

//For Clear Secure Data Base Command
typedef void (*CustomExtConfigClearDBCommand_t)(void);
extern CustomExtConfigClearDBCommand_t CustomExtConfigClearDBCommandCallback;

//For UID Command
typedef void (*CustomExtConfigUidCommand_t)(uint8_t **UID);
extern CustomExtConfigUidCommand_t CustomExtConfigUidCommandCallback;

//For Info Command
typedef void (*CustomExtConfigInfoCommand_t)(uint8_t *Answer);
extern CustomExtConfigInfoCommand_t CustomExtConfigInfoCommandCallback;

//For Help Command
typedef void (*CustomExtConfigHelpCommand_t)(uint8_t *Answer);
extern CustomExtConfigHelpCommand_t CustomExtConfigHelpCommandCallback;

//For PowerStatus Command
typedef void (*CustomExtConfigPowerStatusCommand_t)(uint8_t *Answer);
extern CustomExtConfigPowerStatusCommand_t CustomExtConfigPowerStatusCommandCallback;

//For VersionFw Command
typedef void (*CustomExtConfigVersionFwCommand_t)(uint8_t *Answer);
extern CustomExtConfigPowerStatusCommand_t CustomExtConfigVersionFwCommandCallback;

typedef void (*CustomExtConfigCustomCommand_t)(BLE_CustomCommadResult_t *CustomCommand);
extern CustomExtConfigCustomCommand_t CustomExtConfigCustomCommandCallback;

//For Read Certificate Command
typedef void (*CustomExtConfigReadCertCommand_t)(uint8_t *Certificate);
extern CustomExtConfigReadCertCommand_t CustomExtConfigReadCertCommandCallback;

//For Set Certificate Commnad
typedef void (*CustomExtConfigSetCertCommand_t)(uint8_t *Certificate);
extern CustomExtConfigSetCertCommand_t CustomExtConfigSetCertCommandCallback;

//For Custom Command
typedef void (*CustomExtConfigReadCustomCommands_t)(JSON_Array *JSON_SensorArray);
extern CustomExtConfigReadCustomCommands_t CustomExtConfigReadCustomCommandsCallback;

//For Sensor Configuration
typedef void (*CustomExtConfigReadSensorsConfigCommands_t)(JSON_Array *JSON_SensorArray);
extern CustomExtConfigReadSensorsConfigCommands_t CustomExtConfigReadSensorsConfigCommandsCallback;

typedef void (*CustomExtConfigSetSensorsConfigCommands_t)(uint8_t *Answer);
extern CustomExtConfigSetSensorsConfigCommands_t CustomExtConfigSetSensorsConfigCommandsCallback;

/* Exported functions ------------------------------------------------------- */
extern tBleStatus Stderr_Update(uint8_t *data,uint8_t length);
extern tBleStatus Term_Update(uint8_t *data,uint8_t length);
extern tBleStatus Config_Update(uint32_t Feature,uint8_t Command,uint8_t data);
extern void       setConnectable(void);
extern void       setNotConnectable(void);
extern void       setConnectionParameters(int min , int max, int latency , int timeout );

extern void ResetBleManager(void);
extern tBleStatus InitBleManager(void);
extern int32_t BleManagerAddChar( BleCharTypeDef *BleChar);

extern tBleStatus aci_gatt_update_char_value_wrapper(BleCharTypeDef *BleCharPointer,uint8_t charValOffset,uint8_t charValueLen, uint8_t *charValue);
extern tBleStatus safe_aci_gatt_update_char_value   (BleCharTypeDef *BleCharPointer, uint8_t charValOffset, uint8_t charValueLen, uint8_t *charValue);

extern uint8_t AddCustomCommand(char *CommandName,BLE_CustomCommandTypes_t CommandType, 
                         int32_t Min, int32_t Max, int32_t *ValidValuesInt,char **ValidValuesString,char *ShortDesc,JSON_Array *JSON_SensorArray);
extern void ClearCustomCommandsList(void);

extern void create_JSON_Sensor(COM_Sensor_t *sensor, JSON_Value *tempJSON);

/**
 * @brief  
 * @param  uint8_t* buffer
 * @param  uint8_t len
 * @retval tBleStatus   Status
 */
extern tBleStatus BLE_StdOutSendBuffer(uint8_t* buffer, uint8_t len);

/**
  * @brief  This function is called to parse a BLE_COMM_TP packet.
  * @param  buffer_out: pointer to the output buffer.
  * @param  buffer_in: pointer to the input data.
  * @param  len: buffer in length
  * @retval Buffer out length.
  */
uint32_t BLE_Command_TP_Parse(uint8_t** buffer_out, uint8_t* buffer_in, uint32_t len);

/**
  * @brief  This function is called to prepare a BLE_COMM_TP packet.
  * @param  buffer_out: pointer to the buffer used to save BLE_COMM_TP packet.
  * @param  buffer_in: pointer to the input data.
  * @param  len: buffer in length
  * @retval Buffer out length.
  */
uint32_t BLE_Command_TP_Encapsulate(uint8_t* buffer_out, uint8_t* buffer_in, uint32_t len);

tBleStatus BLE_ExtConfiguration_Update(uint8_t *data,uint32_t length);

#ifdef ACC_BLUENRG_CONGESTION
  #define ACI_GATT_UPDATE_CHAR_VALUE safe_aci_gatt_update_char_value
#else /* ACC_BLUENRG_CONGESTION */
  #define ACI_GATT_UPDATE_CHAR_VALUE aci_gatt_update_char_value_wrapper
#endif /* ACC_BLUENRG_CONGESTION */
#ifdef __cplusplus
}
#endif

#include "BLE_Implementation.h"


/* Control Section -----------------------------------------------------------*/
#ifndef BLE_DEBUG_LEVEL
  #define BLE_DEBUG_LEVEL 3
#else /* BLE_DEBUG_LEVEL */
  #if ((BLE_DEBUG_LEVEL<0) || (BLE_DEBUG_LEVEL>3))
   #error "Valid 0<BLE_DEBUG_LEVEL <3"
  #endif /* Check the define value) */
#endif /* BLE_DEBUG_LEVEL */

#ifndef BLE_MANAGER_USED_PLATFORM
  #error "It's necessary to set the BLE_MANAGER_USED_PLATFORM Compilation Define"
#endif /* BLE_MANAGER_USED_PLATFORM */

#endif /* _BLE_MANAGER_H_ */

/******************* (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
