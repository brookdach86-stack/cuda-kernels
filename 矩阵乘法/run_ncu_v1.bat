@echo off
cd /d "%~dp0"
ncu --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed v1.exe
pause
