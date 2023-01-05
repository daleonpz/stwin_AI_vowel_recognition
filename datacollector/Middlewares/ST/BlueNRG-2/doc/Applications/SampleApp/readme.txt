/**
  @page SampleApp sample application for BlueNRG-2 Expansion Board and STM32 Nucleo Boards
  
  @verbatim
  ******************** (C) COPYRIGHT 2020 STMicroelectronics *******************
  * @file    readme.txt 
  * @author  CL/AST  
  * @version V1.0.0
  * @date    02-Dec-2019
  * @brief   Description of the BlueNRG-2 sample application.
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

This application shows how to simply use the BLE Stack creating a client and server connection. 
It also provides the user with a complete example on how to perform an ATT MTU exchange procedure, 
in order the server and the client can agree on the supported max MTU.

------------------------------------
WARNING: When starting the project from Example Selector in STM32CubeMX and regenerating 
it from ioc file, you may face a build issue. To solve it, remove from the IDE project 
the file stm32l4xx_nucleo.c in Application/User virtual folder and delete from Src and 
Inc folders the files: stm32l4xx_nucleo.c, stm32l4xx_nucleo.h and stm32l4xx_nucleo_errno.h.
------------------------------------

To test this application two STM32 Nucleo boards with their respective X-NUCLEO-BNRG2A1 
STM32 expansion boards should be used. After flashing both the STM32 boards, one board 
configures itself as BLE Server-Peripheral device, while the other as BLE Client-Central 
device. 
After the connection between the two boards is established (signaled by the LD2 LED
blinking on the Client-Central device), pressing the USER button on one board, the
LD2 LED on the other one gets toggled and vice versa.
If you have only one STM32 Nucleo board, you can use the "BLE Scanner" app as BLE
Client-Central device.

The "BLE Scanner" app is available
- for Android at:
https://play.google.com/store/apps/details?id=com.macdom.ble.blescanner
- for iOS at:
https://apps.apple.com/it/app/ble-scanner-4-0/id1221763603

@par Keywords

BLE, Master, Slave, Central, Peripheral, SPI 
 
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
