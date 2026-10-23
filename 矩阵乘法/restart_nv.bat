@echo off
net stop NVDisplay.ContainerLocalSystem /y
timeout /t 3 /nobreak >nul
net start NVDisplay.ContainerLocalSystem
echo driver restarted