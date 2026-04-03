@echo off

title Gemini CLI API

echo =============================================
echo     Starting Gemini CLI API Connection...
echo =============================================
echo. 

cd /d "%~dp0"
node --disable-warning=DEP0040 dist/index.js serve