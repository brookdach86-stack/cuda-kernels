@echo off
REM ============================================================
REM GEMM Nsight Compute 指标采集脚本
REM 用法：右键此文件 -> "以管理员身份运行"
REM 必须以管理员身份运行，否则 ncu 会报 ERR_NVGPUCTRPERM
REM ============================================================

cd /d "D:\AI HPC\CUDA-learning\矩阵乘法"

echo [1/3] 设置注册表 RmProfilingAdminOnly=0 ...
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\AI HPC\CUDA-learning\enable_nsight.ps1"
if errorlevel 1 (
    echo ERROR: 注册表写入失败，请确认已以管理员身份运行！
    pause
    exit /b 1
)

echo [2/3] 重启 NVIDIA 显示驱动以重读配置（屏幕可能闪烁）...
net stop NVDisplay.ContainerLocalSystem /y
timeout /t 3 /nobreak >nul
net start NVDisplay.ContainerLocalSystem
timeout /t 5 /nobreak >nul

echo [3/3] 开始采集 Nsight Compute 指标（8192 尺寸每个约需 1-2 分钟）...

echo === v1 4096 ===
ncu --set full --launch-count 1 -o gemm_v1_4096 .\v1_4096.exe
ncu --import gemm_v1_4096.ncu-rep --csv > gemm_v1_4096.csv 2>&1

echo === v2 4096 ===
ncu --set full --launch-count 1 -o gemm_v2_4096 .\v2_4096.exe
ncu --import gemm_v2_4096.ncu-rep --csv > gemm_v2_4096.csv 2>&1

echo === v3 4096 ===
ncu --set full --launch-count 1 -o gemm_v3_4096 .\v3_4096.exe
ncu --import gemm_v3_4096.ncu-rep --csv > gemm_v3_4096.csv 2>&1

echo === v4 4096 ===
ncu --set full --launch-count 1 -o gemm_v4_4096 .\v4_4096.exe
ncu --import gemm_v4_4096.ncu-rep --csv > gemm_v4_4096.csv 2>&1

echo === v1 8192 ===
ncu --set full --launch-count 1 -o gemm_v1_8192 .\v1_8192.exe
ncu --import gemm_v1_8192.ncu-rep --csv > gemm_v1_8192.csv 2>&1

echo === v2 8192 ===
ncu --set full --launch-count 1 -o gemm_v2_8192 .\v2_8192.exe
ncu --import gemm_v2_8192.ncu-rep --csv > gemm_v2_8192.csv 2>&1

echo === v3 8192 ===
ncu --set full --launch-count 1 -o gemm_v3_8192 .\v3_8192.exe
ncu --import gemm_v3_8192.ncu-rep --csv > gemm_v3_8192.csv 2>&1

echo === v4 8192 ===
ncu --set full --launch-count 1 -o gemm_v4_8192 .\v4_8192.exe
ncu --import gemm_v4_8192.ncu-rep --csv > gemm_v4_8192.csv 2>&1

echo ============================================================
echo ALL DONE. CSV files saved in: D:\AI HPC\CUDA-learning\矩阵乘法
echo ============================================================
pause