@echo off
REM ==========================================================================
REM  Enchaine tout : installation -> publication du code -> evaluation RAGAS
REM  -> publication des resultats. Un journal est ecrit dans
REM  evaluation\results\run_log.txt
REM ==========================================================================
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp01_installer.bat" || goto :fin
call "%~dp03_publier_github.bat" "feat: RAG agentique multi-agents LangGraph (planner, CRAG, Reflexion) + LLM locaux Ollama" || goto :fin
call "%~dp02_evaluer.bat" --gold gold_set.json gold_set2.json || goto :fin
call "%~dp03_publier_github.bat" "eval: resultats RAGAS baseline vs RAG agentique (Ollama)"
:fin
echo.
echo Termine. Appuyer sur une touche pour fermer.
pause >nul
