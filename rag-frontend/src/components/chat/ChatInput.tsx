import { useRef, useState, useCallback, type KeyboardEvent } from 'react'
import { Send, Square, SlidersHorizontal, Upload } from 'lucide-react'
import { useAppDispatch, useAppSelector } from '@/hooks/redux'
import { setMode, selectCurrent } from '@/store/slices/sessionsSlice'
import { toggleSource, selectActiveSources } from '@/store/slices/uiSlice'
import { useChat } from '@/hooks/useChat'
import { cn } from '@/lib/utils'
import { RAG_MODES, DATA_SOURCES } from '@/types'

export function ChatInput() {
  const dispatch   = useAppDispatch()
  const session    = useAppSelector(selectCurrent)
  const sources    = useAppSelector(selectActiveSources)
  const { send, abort, loading } = useChat()
  const [value, setValue] = useState('')
  const [showSources, setShowSources] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const currentMode = RAG_MODES.find((m) => m.id === (session?.mode ?? 'adaptive')) ?? RAG_MODES[0]

  function autoResize() {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
  }

  const handleSend = useCallback(() => {
    if (!value.trim() || loading) return
    send(value.trim())
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [value, loading, send])

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.metaKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const cycleMode = () => {
    if (!session) return
    const idx  = RAG_MODES.findIndex((m) => m.id === session.mode)
    const next = RAG_MODES[(idx + 1) % RAG_MODES.length]
    dispatch(setMode(next.id))
  }

  return (
    <div className="border-t border-border px-6 pt-3.5 pb-5 flex-shrink-0">
      <div className="max-w-[760px] mx-auto">
        <div className="bg-bg-1 border border-border-md rounded-[14px] focus-within:border-accent/50 focus-within:shadow-[0_0_0_3px_rgba(108,143,255,0.08)] transition-all">

          {/* Toolbar */}
          <div className="flex items-center gap-1.5 px-3 pt-2.5">
            {/* Mode selector */}
            <button
              onClick={cycleMode}
              className="flex items-center gap-1.5 text-[11px] px-2 py-1 rounded-sm border transition-all"
              style={{
                color: currentMode.color,
                borderColor: currentMode.color + '55',
                background:  currentMode.color + '18',
              }}
              title={currentMode.description}
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
              </svg>
              {currentMode.label}
            </button>

            {/* Sources toggle */}
            <button
              onClick={() => setShowSources((v) => !v)}
              className={cn(
                'flex items-center gap-1 text-[11px] px-2 py-1 rounded-sm border transition-all',
                showSources
                  ? 'border-accent/55 text-accent bg-accent/18'
                  : 'border-border text-tx-3 hover:border-border-md hover:text-tx-2',
              )}
            >
              <SlidersHorizontal size={11} /> Sources
            </button>

            <button className="flex items-center gap-1 text-[11px] px-2 py-1 rounded-sm border border-border text-tx-3 hover:border-border-md hover:text-tx-2 transition-all">
              <Upload size={11} /> Document
            </button>
          </div>

          {/* Source pills */}
          {showSources && (
            <div className="flex items-center gap-1.5 px-3 pt-1.5 flex-wrap">
              {DATA_SOURCES.map((src) => {
                const active = sources.includes(src.id)
                return (
                  <button
                    key={src.id}
                    onClick={() => dispatch(toggleSource(src.id))}
                    className="flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full border transition-all"
                    style={{
                      color:       active ? src.color : '',
                      borderColor: active ? src.color + '55' : '',
                      background:  active ? src.color + '15' : '',
                    }}
                  >
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: src.color }} />
                    {src.label}
                  </button>
                )
              })}
            </div>
          )}

          {/* Textarea */}
          <div className="flex items-end gap-2 px-3 pt-2 pb-2.5">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => { setValue(e.target.value); autoResize() }}
              onKeyDown={handleKey}
              placeholder="Posez votre question RGPD…"
              rows={1}
              className="flex-1 bg-transparent border-none outline-none resize-none font-sans text-[14px] text-tx placeholder-tx-3 leading-relaxed overflow-y-auto"
              style={{ maxHeight: 140 }}
            />
            <button
              onClick={loading ? abort : handleSend}
              disabled={!loading && !value.trim()}
              className={cn(
                'w-[34px] h-[34px] rounded-[9px] flex items-center justify-center flex-shrink-0 transition-all',
                loading
                  ? 'bg-danger text-white hover:bg-red-600'
                  : value.trim()
                  ? 'bg-accent text-white hover:bg-accent-dark active:scale-[.94]'
                  : 'bg-accent/30 text-white/40 cursor-not-allowed',
              )}
            >
              {loading ? <Square size={13} fill="currentColor" /> : <Send size={13} />}
            </button>
          </div>

          <div className="flex justify-between px-3.5 pb-1.5 text-[10px] text-tx-3">
            <span>Entrée pour envoyer</span>
            <span className="font-mono">⇧↵ nouvelle ligne</span>
          </div>
        </div>
      </div>
    </div>
  )
}
