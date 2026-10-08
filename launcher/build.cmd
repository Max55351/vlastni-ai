@echo off
rem Prelozi spoustec "Vlastni AI.exe" do slozky projektu.
rem Pouziva prekladac C#, ktery je soucasti Windows (.NET Framework 4.x) -
rem nic se nestahuje. Kdyz vedle lezi ikona.ico, zabuduje ji do exe.
cd /d "%~dp0"

set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" (
    echo Prekladac C# nenalezen: %CSC%
    exit /b 1
)

set IKONA=
if exist ikona.ico set IKONA=/win32icon:ikona.ico

"%CSC%" /nologo /codepage:65001 /target:winexe /optimize+ %IKONA% ^
    /reference:System.Windows.Forms.dll ^
    "/out:..\Vlastni AI.exe" VlastniAI.cs
