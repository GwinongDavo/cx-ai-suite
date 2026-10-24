# Wi-Fi keeps disconnecting or is slow
Applies to: Aurora Book 14, Aurora Book Pro 16, Aurora Edge X1

## Common causes
Most Wi-Fi drops come from power-saving settings on the wireless adapter, an outdated Wi-Fi driver, or router band steering between 2.4 GHz and 5 GHz.

## Steps
1. Update the Wi-Fi driver through Aurora Vantage > Updates.
2. In Device Manager, open the wireless adapter > Power Management and untick "Allow the computer to turn off this device to save power".
3. Forget the network and reconnect. If the router supports it, test on the 5 GHz band only.
4. Run `netsh winsock reset` and `ipconfig /flushdns` in an administrator Command Prompt, then restart.

## Hardware check
If the Wi-Fi option is missing completely from Windows, or Device Manager shows no wireless adapter after a BIOS reset, the wireless card may have failed. This is covered under warranty.
