@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo No se encontro Python. Instala Python 3.11 o 3.12 desde python.org.
  pause
  exit /b 1
)
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv
if errorlevel 1 (
  echo No se pudo crear el entorno virtual.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Error instalando dependencias.
  pause
  exit /b 1
)
echo Instalacion completada. Ejecuta INICIAR.bat.
pause
