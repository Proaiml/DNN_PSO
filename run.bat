@echo off
title DNN_PSO - Particle Swarm Optimization for Neural Networks
cls
echo ======================================================================
echo             DNN + PSO: Deep Neural Network Training via PSO
echo ======================================================================
echo.

:: Python 3.11 veya varsayilan Python ile calistir
py -3.11 example.py
if %errorlevel% neq 0 (
    python example.py
)

pause
