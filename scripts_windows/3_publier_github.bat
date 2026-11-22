@echo off
REM ==========================================================================
REM  3) Commit + push sur GitHub (branche agentic-rag par defaut)
REM     3_publier_github.bat "message de commit" [branche]
REM ==========================================================================
chcp 65001 >nul
cd /d "%~dp0.."
setlocal
set MSG=%~1
if "%MSG%"=="" set MSG=feat: RAG agentique multi-agents (LangGraph) avec LLM locaux Ollama
set BRANCH=%~2
if "%BRANCH%"=="" set BRANCH=agentic-rag

for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set CURRENT=%%b
if not "%CURRENT%"=="%BRANCH%" (
  git show-ref --verify --quiet refs/heads/%BRANCH%
  if errorlevel 1 (git checkout -b %BRANCH%) else (git checkout %BRANCH%)
)
git add -A
git diff --cached --quiet && echo Rien a committer. || git commit -m "%MSG%"
git push -u origin %BRANCH%
if errorlevel 1 (
  echo *** Le push a echoue ***
  endlocal & exit /b 1
)
echo Publie sur origin/%BRANCH%.
endlocal
exit /b 0
