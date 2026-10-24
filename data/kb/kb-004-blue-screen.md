# Blue screen (BSOD) after a Windows or driver update
Applies to: all Aurora laptops running Windows 11

## Symptoms
The device restarts with a blue screen showing a stop code such as DRIVER_IRQL_NOT_LESS_OR_EQUAL or INACCESSIBLE_BOOT_DEVICE, often within a day of an update.

## Steps
1. Note the stop code; it decides the next step.
2. If Windows still boots, open Aurora Vantage > Updates and roll back the most recent driver, or uninstall the latest Windows update in Settings > Windows Update > Update history.
3. If Windows does not boot, interrupt startup three times to open Windows Recovery Environment, then choose Troubleshoot > Advanced options > Uninstall Updates.
4. INACCESSIBLE_BOOT_DEVICE after a BIOS update: enter BIOS (press F2 at startup) and confirm the storage mode is set to "NVMe/AHCI", not RAID.

## Escalate when
Escalate to Level 2 if blue screens continue after rolling back updates, or if the hardware diagnostic in Aurora Vantage reports a memory or storage failure. Storage failure with customer data at risk is handled as P2.
