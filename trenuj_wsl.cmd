@echo off
rem Spousti trenink uvnitr WSL. Tohle vola naplanovana uloha po prihlaseni.
rem Okno musi zustat otevrene (i minimalizovane) - dokud bezi, bezi i Linux.
rem
rem   trenuj_wsl.cmd          plny vykon
rem   trenuj_wsl.cmd 0.5      polovicni vykon

set VYKON=%1
if "%VYKON%"=="" set VYKON=1.0

wsl.exe -d Ubuntu-24.04 -- bash -lc "~/vlastni-ai/trenuj_dal.sh %VYKON%"
