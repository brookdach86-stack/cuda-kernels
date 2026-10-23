# 以管理员身份运行此脚本，解锁 Nsight Compute Performance Counter
# 右键 → 使用 PowerShell 运行 或 在管理员 PowerShell 中执行

$gpuKey = Get-ChildItem "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}" -ErrorAction SilentlyContinue |
Where-Object { $_.GetValue("DriverDesc") -like "*NVIDIA*" } |
Select-Object -First 1

if (-not $gpuKey) {
    Write-Host "ERROR: 找不到 NVIDIA GPU 注册表键" -ForegroundColor Red
    exit 1
}

$keyPath = $gpuKey.PSPath.Replace("Microsoft.PowerShell.Core\Registry::", "")
Write-Host "Found GPU at: $keyPath"

reg add $keyPath /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "RmProfilingAdminOnly set to 0" -ForegroundColor Green
    Write-Host "Reboot required. After reboot: ncu .\v1.exe will work." -ForegroundColor Yellow
}
else {
    Write-Host "ERROR: failed to set registry value" -ForegroundColor Red
    exit 1
}