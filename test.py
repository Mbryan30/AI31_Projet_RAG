import os
import reponse  # ton module reponse.py

# Active les couleurs ANSI sur Windows
os.system("")

"""
Quelles sont les bases légales pour traiter des données personnelles ?
Dans quel chapitre se trouve l'article 45 et l'article 56 ?
Quels sont les droits d'une personne concernée par un traitement de données ?
c'est quel chapitre qui parle de Dispositions relatives à des situations particulières de traitement ?

"""

# Palette
CYAN, GREEN, YELLOW, MAGENTA, GRAY, RED = (
    "\033[96m", "\033[92m", "\033[93m", "\033[95m", "\033[90m", "\033[91m"
)
BOLD, RESET = "\033[1m", "\033[0m"


def banner():
    print(f"\n{CYAN}{BOLD}{'═' * 62}{RESET}")
    print(f"{CYAN}{BOLD}   🛡️   ASSISTANT RGPD — Interface de test{RESET}")
    print(f"{CYAN}{BOLD}{'═' * 62}{RESET}")
    print(f"{GRAY}   Tape ta question (ou 'exit' pour quitter).{RESET}\n")


def afficher_reponse(resultat):
    print(f"\n{GREEN}{BOLD}┌─ RÉPONSE {'─' * 51}{RESET}")
    for ligne in resultat["reponse"].splitlines():
        print(f"{GREEN}│{RESET}  {ligne}")
    print(f"{GREEN}{BOLD}└{'─' * 61}{RESET}")

    if resultat["articles_cites"]:
        print(f"\n{YELLOW}{BOLD}📚 Articles cités :{RESET}")
        for a in resultat["articles_cites"]:
            print(f"{YELLOW}   • {a}{RESET}")
    print()


def main():
    banner()
    while True:
        try:
            question = input(f"{MAGENTA}{BOLD}❯ Question : {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        print(f"{GRAY}   … recherche en cours …{RESET}")
        try:
            resultat = reponse.smart_rag_pipeline(question)
            afficher_reponse(resultat)
        except Exception as e:
            print(f"\n{RED}⚠  Erreur : {e}{RESET}\n")

    print(f"\n{CYAN}À bientôt 👋{RESET}\n")


if __name__ == "__main__":
    main()
