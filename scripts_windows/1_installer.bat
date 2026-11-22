@echo off
REM ==========================================================================
REM  1) Installe les dependances Python du RAG agentique et les modeles Ollama
REM ==========================================================================
chcp 65001 >nul
cd /d "%~dp0.."
setlocal

if not defined LLM_MODEL set LLM_MODEL=qwen2.5:7b
if not defined JUDGE_MODEL set JUDGE_MODEL=qwen2.5:14b

set PY=python
if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe

echo [1/3] Dependances Python (agentique + Ollama)...
%PY% -m pip install --upgrade "langgraph>=1.0,<2.0" "langchain-ollama>=1.0" "rank-bm25>=0.2.2" "matplotlib>=3.8"
if errorlevel 1 goto :error

echo [2/3] Verification d'Ollama...
where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama introuvable : installer depuis https://ollama.com/download puis relancer.
  goto :error
)
ollama list

echo [3/3] Modeles : %LLM_MODEL% (RAG) et %JUDGE_MODEL% (juge RAGAS)
ollama list | findstr /B /C:"%LLM_MODEL% " >nul || ollama pull %LLM_MODEL%
if errorlevel 1 goto :error
ollama list | findstr /B /C:"%JUDGE_MODEL% " >nul || ollama pull %JUDGE_MODEL%
if errorlevel 1 goto :error

echo.
echo Installation terminee.
endlocal
exit /b 0

:error
echo.
echo *** Erreur pendant l'installation ***
endlocal
exit /b 1
