@echo off
rem ---------------------------------------------------------------------
rem  Samostatny beh treninku - nezavisly na Claude Code i na prihlaseni.
rem  Kdyz trenink spadne (dojde pamet grafiky, ovladac se zasekne),
rem  tenhle davkovy soubor ho do minuty nastartuje znovu od posledni zalohy.
rem
rem  Spusteni:  trenuj_dal.cmd            (plny vykon)
rem             trenuj_dal.cmd 0.5        (polovicni vykon)
rem  Zastaveni: zavri okno, nebo Ctrl+C
rem ---------------------------------------------------------------------

cd /d "%~dp0"

rem Pojistka proti dvojimu spusteni: dva treninky najednou by se poprali
rem o pamet grafiky a spadly by oba. Kdyz uz nejaky bezi, tenhle skonci.
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*train.py*' }) { exit 1 } else { exit 0 }"
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] trenink uz bezi, tenhle se nespousti
    ping -n 11 127.0.0.1 > nul
    exit /b 0
)

set VYKON=%1
if "%VYKON%"=="" set VYKON=1.0

set PY=%~dp0.venv\Scripts\python.exe
set POKUS=0
rem kolik padu po sobe bez jakehokoli postupu - pri trvale chybe
rem (treba Windows zablokuje knihovnu) nema smysl zkouset donekonecna
set MARNE=0
set MAX_MARNYCH=5

:znovu
echo.
echo [%date% %time%] spoustim trenink (vykon %VYKON%, restartu zatim %POKUS%)
echo.

set PRED=
for %%f in (out\model_last.pt) do set PRED=%%~tf

"%PY%" train.py --resume --vykon %VYKON% --batch-size 3 >> trenink.log 2>&1
rem kod chyby ulozit hned - kazdy dalsi prikaz (i set /a) ho prepise
set KOD=%ERRORLEVEL%

rem navratovy kod 0 = trenink dobehl do konce, jinak spadl
if %KOD% EQU 0 goto hotovo

set /a POKUS=%POKUS%+1

rem Postup se pozna podle toho, ze se mezitim ulozila nova zaloha.
set PO=
for %%f in (out\model_last.pt) do set PO=%%~tf
if "%PRED%"=="%PO%" (set /a MARNE=%MARNE%+1) else (set MARNE=0)

if %MARNE% GEQ %MAX_MARNYCH% (
    echo [%date% %time%] %MARNE% padu po sobe bez postupu, posledni chyba %KOD% - vzdavam to, podivej se do trenink.log >> trenink.log
    echo [%date% %time%] trenink opakovane pada bez postupu - konec. Detail v trenink.log
    ping -n 61 127.0.0.1 > nul
    exit /b %KOD%
)

echo [%date% %time%] trenink skoncil s chybou %KOD%, restart za 60 s (marnych pokusu %MARNE%/%MAX_MARNYCH%) >> trenink.log
rem ping misto timeout - timeout selze, kdyz skript bezi bez konzole
rem (naplanovana uloha), ping ceka spolehlive vzdycky
ping -n 61 127.0.0.1 > nul
goto znovu

:hotovo
echo.
echo [%date% %time%] TRENINK DOKONCEN
echo [%date% %time%] TRENINK DOKONCEN >> trenink.log
rem zadny "pause" - v naplanovane uloze neni kdo by stiskl klavesu
rem a uloha by visela navzdy. Okno zustane minutu otevrene a zavre se.
ping -n 61 127.0.0.1 > nul
