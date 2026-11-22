@echo off
REM ==========================================================================
REM  2) Evaluation RAGAS comparative : RAG avance (baseline) vs RAG agentique
REM     Les arguments sont transmis au script, ex. :
REM       2_evaluer.bat --limit 3
REM       2_evaluer.bat --gold gold_set.json gold_set2.json
REM     Le script est reprenable : relancer reprend la ou il s'etait arrete.
REM ==========================================================================
chcp 65001 >nul
cd /d "%~dp0.."
set PY=python
if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
set PYTHONUTF8=1
%PY% -u evaluation\evaluation_comparative.py %*
exit /b %errorlevel%
