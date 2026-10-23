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

Set-ItemProperty -Path $keyPath -Name "RmProfilingAdminOnly" -Value 0 -Type DWord -Force
Write-Host "RmProfilingAdminOnly set to 0" -ForegroundColor Green
Write-Host "Reboot required. After reboot: ncu .\v1.exe will work." -ForegroundColor Yellow