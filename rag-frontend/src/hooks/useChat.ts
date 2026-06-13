import { useCallback, useRef, useState } from 'react'
import { v4 as uuid } from 'uuid'
import { useAppDispatch, useAppSelector } from './redux'
import {
  addMessage,
  updateStreamingMessage,
  finalizeMessage,
  selectCurrent,
} from '@/store/slices/sessionsSlice'
import { selectActiveSources } from '@/store/slices/uiSlice'
import { showToast } from '@/store/slices/uiSlice'
import { sendQuery } from '@/lib/api'
import type { Message } from '@/types'

export function useChat() {
  const dispatch   = useAppDispatch()
  const session    = useAppSelector(selectCurrent)
  const sources    = useAppSelector(selectActiveSources)
  const [loading, setLoading] = useState(false)
  const abortRef   = useRef<AbortController | null>(null)

  const send = useCallback(
    async (text: string) => {
      if (!session || !text.trim() || loading) return

      // 1. Push user message
      const userMsg: Message = {
        id:      uuid(),
        role:    'user',
        content: text.trim(),
        ts:      Date.now(),
      }
      dispatch(addMessage(userMsg))

      // 2. Create placeholder assistant message
      const assistantId = uuid()
      const placeholder: Message = {
        id:          assistantId,
        role:        'assistant',
        content:     '',
        ts:          Date.now(),
        isStreaming: true,
      }
      dispatch(addMessage(placeholder))
      setLoading(true)

      try {
        abortRef.current = new AbortController()

        // ── Replace with streamQuery() for SSE streaming ──────────
        const result = await sendQuery({
          question:   text.trim(),
          session_id: session.id,
          mode:       session.mode,
          sources:    sources as any,
        })

        // Simulate token-by-token for demo (remove when using real streaming)
        for (const word of result.answer.split(' ')) {
          dispatch(updateStreamingMessage({ id: assistantId, delta: word + ' ' }))
          await new Promise((r) => setTimeout(r, 18))
        }

        dispatch(
          finalizeMessage({
            id:      assistantId,
            sources: result.sources,
            metrics: result.metrics,
          }),
        )
      } catch (err: unknown) {
        if ((err as Error).name === 'AbortError') return
        dispatch(showToast({ icon: '⚠', message: 'Erreur pipeline RAG' }))
        // Remove placeholder on error
        dispatch(
          finalizeMessage({ id: assistantId, sources: [], metrics: { latency: 0, tokens: 0, strategy: '—' } }),
        )
      } finally {
        setLoading(false)
      }
    },
    [dispatch, session, sources, loading],
  )

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setLoading(false)
  }, [])

  return { send, abort, loading }
}
