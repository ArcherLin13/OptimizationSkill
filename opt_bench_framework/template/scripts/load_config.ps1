param([hashtable]$Vars)

$localConfig = Join-Path $PSScriptRoot "config.local.ps1"
if (Test-Path $localConfig) { . $localConfig }

if ($Vars.ContainsKey("OhosNative") -and [string]::IsNullOrWhiteSpace($Vars.OhosNative)) {
    if ($env:OHOS_NATIVE) { $Vars.OhosNative = $env:OHOS_NATIVE }
    elseif ($script:OHOS_NATIVE) { $Vars.OhosNative = $script:OHOS_NATIVE }
}
if ($Vars.ContainsKey("OpenCvRoot") -and [string]::IsNullOrWhiteSpace($Vars.OpenCvRoot)) {
    if ($script:OPENCV_ROOT) { $Vars.OpenCvRoot = $script:OPENCV_ROOT }
}
if ($Vars.ContainsKey("DeviceLibDir") -and [string]::IsNullOrWhiteSpace($Vars.DeviceLibDir)) {
    if ($script:DEVICE_LIB_DIR) { $Vars.DeviceLibDir = $script:DEVICE_LIB_DIR }
}
if ($Vars.ContainsKey("RemoteDir") -and [string]::IsNullOrWhiteSpace($Vars.RemoteDir)) {
    if ($script:REMOTE_DIR) { $Vars.RemoteDir = $script:REMOTE_DIR }
}
return $Vars
