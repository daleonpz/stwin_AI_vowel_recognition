/**
  @page Beacon sample application for BlueNRG-2 Expansion Board and STM32 Nucleo Boards
  
  @verbatim
  ******************** (C) COPYRIGHT 2020 STMicroelectronics *******************
  * @file    readme.txt  
  * @author  MCD & CL/AST
  * @version V1.0.0
  * @date    02-Dec-2019
  * @brief   This application contains an example which shows how implementing
  *          an iBeacon device.
  *          The communication is done using a STM32 Nucleo board and a Smartphone
  *          with BLE.
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

This application shows how to use the BlueNRG-2 STM32 expansion board to 
implement a Beacon device. 

------------------------------------
WARNING: When starting the project from Example Selector in STM32CubeMX and regenerating 
it from ioc file, you may face a build issue. To solve it, remove from the IDE project 
the file stm32l4xx_nucleo.c in Application/User virtual folder and delete from Src and 
Inc folders the files: stm32l4xx_nucleo.c, stm32l4xx_nucleo.h and stm32l4xx_nucleo_errno.h.
------------------------------------

A Beacon device is a smart Bluetooth Low Energy device that transmits a small data 
payload at regular intervals using Bluetooth advertising packets.
Beacons are used to mark important places and objects. Typically, a beacon is 
visible to a user's device from a range of a few meters, allowing for highly 
context-sensitive use cases.
To locate the beacon, it is necessary to have a scanner application running on a 
BLE-capable smartphone, such as "BLE Scanner" app.
The "BLE Scanner" app is available
- for Android at:
  https://play.google.com/store/apps/details?id=com.macdom.ble.blescanner
- for iOS at:
  https://apps.apple.com/it/app/ble-scanner-4-0/id1221763603

@par Keywords

BLE, Beacon, SPI
  
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
 - Rebuild all files and load your image into target memory.
 - Run the example.
 - Alternatively, you can download the pre-built binaries in "Binary" 
   folder included in the distributed package.


 * <h3><center>&copy; COPYRIGHT STMicroelectronics</center></h3>
 */
