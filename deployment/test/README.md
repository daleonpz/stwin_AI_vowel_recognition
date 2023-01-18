# Test Environment

Hardware: SOLUCON Gateway V2 Rev. 8 / 9

Software: moehlenhoff-gateway v0.8.1, kernel cc1101 driver v2.0.5

Testdate: 26.04.2018

# Test Data

## Gateway Config

- moehlenhoff-gateway.cfg

## cSP-L packets

send packets using the cc1101-driver test tool

`test_cc1101 1 <filesize> 1 0 <filename>`

- cSP-L_01_ftc_payload_w_feedback.bin

Gateway will return a non-ftc feedback packet

- cSP-L_02_ftc_get_firmware.bin

Gateway will return its gateway id in an non-ftc firmware packet

- cSP-L_03_non-ftc_get_time_request.bin

Gateway will return a non-ftc time packet

- cSP-L_04_non-ftc_get_time_response.bin

Gateway will discard packet

- cSP-L_05_non-ftc_get_firmware_chunk_0.bin

Gateway will return a non-ftc payload without feedback packet with full chunk 0

- cSP-L_06_non-ftc_get_firmware_chunk_1.bin

Gateway will return a non-ftc payload without feedback packet with full chunk 1

- cSP-L_07_non-ftc_get_firmware_chunk_A.bin

Gateway will return a non-ftc payload without feedback packet with last chunk A

- cSP-L_08_non-ftc_get_firmware_chunk_F.bin

Gateway will discard packet, chunk out of range

## web server data

copy image to the following path and rename it to image

`devices/3e183e8d/firmwares/df2114e8-2e6c-42c3-a09e-5a262453d0e8`

- 3e183e8d.bin

## mqtt json input

send json files to mqtt broker

`mosquitto_pub -d -h 10.13.37.250 -q 1 -t "iot-gateway/solucon-gateway/firmware/update" -f <filename>`

- mqtt_01_firmware-update-device.json

Gateway will download a device firmware from webserver and store the device it

- mqtt_02_firmware-update-gateway.json

Gateway will download its firmware and flash it

# Generate more test data

- generate-json_firmware-update-device.sh
- generate-json_firmware-update-gateway.sh
- generate-packet_cSP-L_gw-in.sh

# Further test software

- kernel-driver_test_burst_all_packet_sizes.c
- kernel-driver_test_burst_cSP-L_packet_size.c
- kernel-driver_test_switch_freq.c
- unit-test_wakeup-packet-generator.c
