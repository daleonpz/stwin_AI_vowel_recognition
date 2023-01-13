/**
  @page SensorDemo_BLESensor-App sample application for BlueNRG-2 Expansion Board and STM32 Nucleo Boards
  
  @verbatim
  ******************** (C) COPYRIGHT 2020 STMicroelectronics *******************
  * @file    readme.txt  
  * @author  CL/AST
  * @version V1.0.0
  * @date    02-Dec-2019
  * @brief   This application contains an example which shows how implementing
  *          proprietary Bluetooth Low Energy profiles.
  *          The communication is done using a STM32 Nucleo board and a Smartphone
  *          with BTLE.
  ******************************************************************************
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  @endverbatim

  
@par Example Description 

This application shows how to implement a peripheral device tailored for 
interacting with the "ST BLE Sensor" app for Android/iOS devices.

------------------------------------
WARNING: When starting the project from Example Selector in STM32CubeMX and regenerating 
it from ioc file, you may face a build issue. To solve it, remove from the IDE project 
the file stm32l4xx_nucleo.c in Application/User virtual folder and delete from Src and 
Inc folders the files: stm32l4xx_nucleo.c, stm32l4xx_nucleo.h and stm32l4xx_nucleo_errno.h.
------------------------------------

The "ST BLE Sensor" app is freely available on both GooglePlay and iTunes
  - iTunes: https://itunes.apple.com/us/app/st-bluems/id993670214
  - GooglePlay: https://play.google.com/store/apps/details?id=com.st.bluems
The source code of the "ST BLE Sensor" app is available on GitHub at:
  - iOS: https://github.com/STMicroelectronics-CentralLabs/STBlueMS_iOS
  - Android: https://github.com/STMicroelectronics-CentralLabs/STBlueMS_Android

@note: NO SUPPORT WILL BE PROVIDED TO YOU BY STMICROELECTRONICS FOR ANY OF THE
ANDROID/iOS app INCLUDED IN OR REFERENCED BY THIS PACKAGE.

After establishing the connection between the STM32 board and the smartphone:
  - the emulated values of temperature and pressure are sent by the STM32 board to 
    the mobile device and are shown in the ENVIRONMENTAL tab;
  - the emulated sensor fusion data sent by the STM32 board to the mobile device 
    reflects into the cube rotation showed in the app's MEMS SENSOR FUSION tab
  - the plot of the emulated data (temperature, pressure, sensor fusion, 
    accelerometer, gyroscope and magnetometer), sent by the STM32 board, are shown in the 
	PLOT DATA tab;
  - in the RSSI & Battery tab the RSSI value is shown.
According to the value of the #define USE_BUTTON in file app_bluenrg_2.c, the 
environmental and the motion data can be sent automatically (with 1 sec period) 
or when the User Button is pressed.

The communication is done using a vendor specific profile.

-----------------------------------------------------
||||||||||||| VERY IMPORTANT NOTES for FOTA Feature |
|||||||||||||        (only for STM32L476RG)         |
-----------------------------------------------------
 1) For the STM32L476RG MCU, this example support the Firmware-Over-The-Air (FOTA) 
    update using the ST BLE Sensor Android/iOS application (Version 3.0.0 or higher) 
 
 2) This example must run starting at address 0x08004000 in memory and works ONLY if the BootLoader 
    is saved at the beginning of the FLASH (address 0x08000000)
 	
 3) When generating a SensorDemo_BLESensorApp project for the STM32L476RG MCU with the STM32CubeMX, to 
    correctly run the FOTA feature, the following modifications to the code and the project are required 
	before building:

	3.1) In file Src/system_stm32l4xx.c, enable the #define USER_VECT_TAB_ADDRESS and set the VECT_TAB_OFFSET 
	     to 0x4000 (the default value is 0x00)
	     #define VECT_TAB_OFFSET  0x4000 /*!< Vector Table base offset field.
                                              This value must be a multiple of 0x200. */
											
    3.2) In your IDE, set the .intvec and ROM start addresses to 0x08004000, following the instructions below:
	     - EWARM
	       Project -> Options --> Linker --> Config --> Edit
            - Vector Table --> .intvec start 0x08004000
            - Memory Regions --> ROM: Start 0x08004000 - End 0x080FFFFF
         - MDK-ARM 
	       Project -> Options -> Target 
            - IROM1: 0x8004000
         - STM32CubeIDE
	       Open linker script file STM32L476RGTX_FLASH.ld and set the Flash origin address to:
           /* Memories definition */
           MEMORY
           {
             RAM    (xrw)    : ORIGIN = 0x20000000,   LENGTH = 96K
             RAM2    (xrw)    : ORIGIN = 0x10000000,   LENGTH = 32K
             FLASH    (rx)    : ORIGIN = 0x8004000,   LENGTH = 1024K
           }

 4) For debugging activity (Optimizations=None), increase the value of cstack (e.g. from 0x400 to 0x800)
    and heap (e.g. from 0x200 to 0x400)
	
 5) For each IDE (IAR/Keil uVision/STM32CubeIDE) a *.bat/*.sh script file (in folder GenSensorDemoBin_Script 
    contained in the archive Utilities\BootLoader\BootLoader.zip) must be used for running the following 
	operations on the STM32L476RG MCU:
    - Full Flash Erase
    - Load the BootLoader (in folder STM32L476RG_BL contained in the archive Utilities\BootLoader\BootLoader.zip) 
	  on the correct flash region
    - Load the Program (after the compilation) on the correct flash region (this could be used for a FOTA)
    - Dump back one single binary that contain BootLoader+Program that could be 
      flashed at the flash beginning (address 0x08000000) (this COULD BE NOT used for FOTA)
    - Reset the board
	To easily use these scripts, extract the BootLoader.zip archive in the root folder of your 
	SensorDemo_BLESensor-App sample application.
    Before launching the script for your IDE, open it and set the correct paths and filenames.


-----------------------------------------------------
|||||||||||||||||||||||||||||||||||||||||||||||||||||
-----------------------------------------------------

@par Keywords

BLE, Peripheral, SPI
  
@par Hardware and Software environment

  - This example runs on STM32 Nucleo devices with BlueNRG-2 STM32 expansion board
    (X-NUCLEO-BNRG2A1)
  - This example has been tested with STMicroelectronics:
    - NUCLEO-L476RG RevC board

  
@par How to use it? 

In order to make the program work, you must do the following:
 - WARNING: before opening the project with any toolchain be sure your folder
   installation path is not too in-depth since the toolchain may report errors
   after building.
 - Open STM32CubeIDE (this firmware has been successfully tested with Version 1.5.1).
   Alternatively you can use the Keil uVision toolchain (this firmware
   has been successfully tested with V5.31.0) or the IAR toolchain (this firmware has 
   been successfully tested with Embedded Workbench V8.50.5).
 - Rebuild all files and 
   - for STM32L476RG MCU: run the *.bat/*.sh script included in the IDE folder 
     (EWARM/MDK-ARM/STM32CubeIDE)
   - for other STM32 MCU: load your image into target memory
 - Run the example.
 - Alternatively, you can download the pre-built *.bin in "Binary" folder
   included in the distributed package (for STM32L476RG MCU, the *_BL.bin file).


 * <h3><center>&copy; COPYRIGHT STMicroelectronics</center></h3>
 */
