@echo off
echo ============================================
echo   vLLM Manager - Build EXE
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato nel PATH.
    echo Installa Python da https://python.org
    pause
    exit /b 1
)

:: Install PyInstaller if needed
echo [1/3] Verifico PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installazione PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERRORE] Impossibile installare PyInstaller.
        pause
        exit /b 1
    )
)

:: Build
echo [2/3] Compilazione in corso...
echo.
pyinstaller --onefile --windowed --name "vLLM Manager" vllm_manager.py

if errorlevel 1 (
    echo.
    echo [ERRORE] Compilazione fallita.
    pause
    exit /b 1
)

:: Done
echo.
echo [3/3] Compilazione completata!
echo.
echo Output: dist\vLLM Manager.exe
echo.
echo Puoi copiare "dist\vLLM Manager.exe" dove preferisci.
echo.
pause
