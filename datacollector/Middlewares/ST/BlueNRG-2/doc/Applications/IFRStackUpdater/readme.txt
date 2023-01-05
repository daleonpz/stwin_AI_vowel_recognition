/**
  @page IFRStack_Updater sample application for BlueNRG-2 Expansion Board and STM32 Nucleo Boards
  
  @verbatim
  ******************** (C) COPYRIGHT 2020 STMicroelectronics *******************
  * @file    readme.txt  
  * @author  CL/AST  
  * @version V1.0.0
  * @date    30-Sept-2020
  * @brief   This application is an example to be loaded in order to update the
  *          BlueNRG-2 fw stack to latest version or to change the IFR
  *          configuration.
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

This application should be used to update the BlueNRG-2 firmware stack to latest version 
or to change the IFR configuration.

------------------------------------
WARNING: When starting the project from Example Selector in STM32CubeMX and regenerating 
it from ioc file, you may face a build issue. To solve it, remove from the IDE project 
the file stm32l4xx_nucleo.c in Application/User virtual folder and delete from Src and 
Inc folders the files: stm32l4xx_nucleo.c, stm32l4xx_nucleo.h and stm32l4xx_nucleo_errno.h.
------------------------------------

To change the IFR configuration the #define BLUENRG_DEV_CONFIG_UPDATER must be defined.
The IFR parameters can be changed editing the file Middlewares\ST\BlueNRG-2\hci\bluenrg1_devConfig.c.

@note: Be sure you know what you are doing when modifying the IFR settings.
       This operation should be performed only by expert users. 

To update the BlueNRG-2 fw stack the #define BLUENRG_STACK_UPDATER must be
defined (see file app_bluenrg_2.h).
The FW image used by the application is contained in the FW_IMAGE array defined in 
file update_fw_image.c. In the same file the DTM SPI and FW versions are reported.

For some STM32 MCUs with low flash size, when the BLUENRG_STACK_UPDATER is defined, 
a linking error may be faced. 
For such STM32 MCUs, the DTM image can be updated using a serial terminal (e.g. HyperTerminal 
or TeraTerm) and its transfer feature based on the YMODEM.
To enable this configuration the #define BLUENRG_STACK_UPDATER_YMODEM must be defined 
in file app_bluenrg_2.h (defined by default).
In this configuration, if for instance the TeraTerm is used, after loading the binary file 
on the STM32:
 - open a connection 
   Speed: 115200, Data: 8 bit, Parity: None, Stop bits: 1, Flow control: none
   New-line: Receive=AUTO, Transmit=CR
 - go to File -> Transfer -> YMODEM -> Send
 - select the DTM SPI binary file to load from the PC's file system
   (e.g. the DTM_SPI_NOUPDATER.bin contained in the installation folder of the ST's BlueNRG GUI
   C:\Users\<username>\ST\BlueNRG GUI X.Y.Z\Firmware\BlueNRG2\DTM)

After loading the IFR/Stack Updater application on a STM32 Nucleo board equipped 
with a X-NUCLEO-BNRG2A1 expansion board: 
  - the LD2 LED on indicates that the updating process has started and is running
  - the slowly blinking LD2 LED (T = 3 secs) indicates that the updating process 
    has failed
  - the fast blinking LD2 LED (T = 0.5 secs) indicates that the updating process 
    has successfully terminated

This application project requires: 
  - CSTACK minimum size 0xD00
  - HEAP minimum size 0x200

@par Keywords

BLE, SPI, YMODEM
  
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
