import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import { v4 as uuid } from 'uuid'
import type { Session, Message, RagMode, SourceId } from '@/types'

// ─── Persist helpers ─────────────────────────────────────────
const KEY = 'rag_sessions_v3'

function persist(sessions: Session[]) {
  try { localStorage.setItem(KEY, JSON.stringify(sessions)) } catch { /* quota */ }
}

function hydrate(): Session[] {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as Session[]) : []
  } catch { return [] }
}

// ─── State ───────────────────────────────────────────────────
interface SessionsState {
  sessions: Session[]
  currentId: string | null
}

const initialState: SessionsState = {
  sessions:  hydrate(),
  currentId: hydrate()[0]?.id ?? null,
}

// ─── Slice ───────────────────────────────────────────────────
export const sessionsSlice = createSlice({
  name: 'sessions',
  initialState,
  reducers: {
    createSession(state) {
      const session: Session = {
        id:       uuid(),
        title:    'Nouvelle session',
        messages: [],
        created:  Date.now(),
        updated:  Date.now(),
        mode:     'adaptive',
      }
      state.sessions.unshift(session)
      state.currentId = session.id
      persist(state.sessions)
    },

    switchSession(state, action: PayloadAction<string>) {
      state.currentId = action.payload
    },

    renameSession(state, action: PayloadAction<{ id: string; title: string }>) {
      const s = state.sessions.find((x) => x.id === action.payload.id)
      if (s) { s.title = action.payload.title; s.updated = Date.now() }
      persist(state.sessions)
    },

    deleteSession(state, action: PayloadAction<string>) {
      state.sessions = state.sessions.filter((s) => s.id !== action.payload)
      if (state.currentId === action.payload) {
        state.currentId = state.sessions[0]?.id ?? null
      }
      persist(state.sessions)
    },

    clearMessages(state) {
      const s = state.sessions.find((x) => x.id === state.currentId)
      if (s) { s.messages = []; s.updated = Date.now() }
      persist(state.sessions)
    },

    addMessage(state, action: PayloadAction<Message>) {
      const s = state.sessions.find((x) => x.id === state.currentId)
      if (!s) return
      // Auto-title from first user message
      if (s.messages.length === 0 && action.payload.role === 'user') {
        s.title = action.payload.content.length > 42
          ? action.payload.content.slice(0, 42) + '…'
          : action.payload.content
      }
      s.messages.push(action.payload)
      s.updated = Date.now()
      persist(state.sessions)
    },

    updateStreamingMessage(state, action: PayloadAction<{ id: string; delta: string }>) {
      const s = state.sessions.find((x) => x.id === state.currentId)
      if (!s) return
      const msg = s.messages.find((m) => m.id === action.payload.id)
      if (msg) msg.content += action.payload.delta
    },

    finalizeMessage(
      state,
      action: PayloadAction<{
        id: string
        sources: Message['sources']
        metrics: Message['metrics']
      }>,
    ) {
      const s = state.sessions.find((x) => x.id === state.currentId)
      if (!s) return
      const msg = s.messages.find((m) => m.id === action.payload.id)
      if (msg) {
        msg.isStreaming = false
        msg.sources     = action.payload.sources
        msg.metrics     = action.payload.metrics
      }
      s.updated = Date.now()
      persist(state.sessions)
    },

    setMode(state, action: PayloadAction<RagMode>) {
      const s = state.sessions.find((x) => x.id === state.currentId)
      if (s) s.mode = action.payload
      persist(state.sessions)
    },
  },
})

export const {
  createSession,
  switchSession,
  renameSession,
  deleteSession,
  clearMessages,
  addMessage,
  updateStreamingMessage,
  finalizeMessage,
  setMode,
} = sessionsSlice.actions

export default sessionsSlice.reducer

// ─── Selectors ───────────────────────────────────────────────
import type { RootState } from '../index'

export const selectSessions   = (s: RootState) => s.sessions.sessions
export const selectCurrentId  = (s: RootState) => s.sessions.currentId
export const selectCurrent    = (s: RootState) =>
  s.sessions.sessions.find((x) => x.id === s.sessions.currentId) ?? null
