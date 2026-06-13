# RAG·AI Frontend

Interface React + Vite + Redux Toolkit + Tailwind pour votre pipeline RAG RGPD.

## Stack

| Couche | Techno |
|---|---|
| Framework | React 18 + Vite 5 |
| Langage | TypeScript strict |
| État global | Redux Toolkit + persist localStorage |
| Style | Tailwind CSS v3 (design system custom dark) |
| Routing | React Router v6 |
| Markdown | react-markdown + remark-gfm |
| HTTP | Axios + proxy Vite vers FastAPI |
| Streaming | EventSource (SSE) prêt à l'emploi |
| Deploy | Vercel (vercel.json inclus) |

## Structure

```
src/
├── components/
│   ├── chat/
│   │   ├── ChatArea.tsx        # Liste des messages + scroll auto
│   │   ├── ChatInput.tsx       # Input avec mode RAG + sources
│   │   ├── MessageBubble.tsx   # Bulle message (Markdown, sources, métriques)
│   │   └── WelcomeScreen.tsx   # Écran vide avec suggestions
│   ├── layout/
│   │   ├── TopBar.tsx          # Barre du haut (titre, export, clear)
│   │   └── RenameModal.tsx     # Modal rename session
│   ├── sidebar/
│   │   ├── Sidebar.tsx         # Sidebar complète
│   │   └── SessionItem.tsx     # Item de session avec actions
│   └── ui/
│       ├── Button.tsx          # Composant Button (primary/ghost/icon)
│       ├── Modal.tsx           # Modal générique
│       └── Toast.tsx           # Notification toast
├── hooks/
│   ├── redux.ts                # useAppDispatch / useAppSelector typés
│   ├── useChat.ts              # Orchestration envoi + streaming
│   └── useToast.ts             # Helper toast
├── lib/
│   ├── api.ts                  # Client Axios + sendQuery + streamQuery SSE
│   └── utils.ts                # cn(), timeAgo(), formatDate(), sleep()
├── store/
│   ├── index.ts                # Store Redux
│   └── slices/
│       ├── sessionsSlice.ts    # Sessions + messages + persist
│       └── uiSlice.ts          # Sidebar, modal, toast, sources actives
└── types/
    └── index.ts                # Types TS + constantes RAG_MODES, DATA_SOURCES
```

## Installation

```bash
npm install
cp .env.example .env          # Configurer VITE_API_URL
npm run dev
```

## Connecter votre API FastAPI

Dans `src/lib/api.ts`, `sendQuery()` pointe vers `POST /api/query`.

Votre endpoint doit retourner :

```json
{
  "answer": "...",
  "sources": [{ "title": "...", "score": 97 }],
  "metrics": { "latency": 420, "tokens": 280, "strategy": "CRAG" }
}
```

Pour le **streaming SSE**, utilisez `streamQuery()` dans `useChat.ts` à la place de `sendQuery()`.

## Déploiement Vercel

```bash
npm run build
# Push sur GitHub → connecter repo sur vercel.com
# Ajouter VITE_API_URL dans les variables d'environnement Vercel
```

Le fichier `vercel.json` gère le rewrite SPA et le cache des assets.
