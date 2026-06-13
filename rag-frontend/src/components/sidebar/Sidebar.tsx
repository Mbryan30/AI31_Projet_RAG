import { useState } from 'react'
import { Plus, Search } from 'lucide-react'
import { useAppDispatch, useAppSelector } from '@/hooks/redux'
import { useToast } from '@/hooks/useToast'
import {
  selectSessions, selectCurrentId,
  createSession, switchSession, deleteSession,
} from '@/store/slices/sessionsSlice'
import { openRenameModal, selectSidebarOpen } from '@/store/slices/uiSlice'
import { cn } from '@/lib/utils'
import { SessionItem } from './SessionItem'

export function Sidebar() {
  const dispatch  = useAppDispatch()
  const sessions  = useAppSelector(selectSessions)
  const currentId = useAppSelector(selectCurrentId)
  const sidebarOpen = useAppSelector(selectSidebarOpen)
  const { toast } = useToast()
  const [query, setQuery] = useState('')

  const filtered = sessions.filter((s) =>
    s.title.toLowerCase().includes(query.toLowerCase()),
  )

  const totalMessages = sessions.reduce((a, s) => a + s.messages.length, 0)

  function handleNew() {
    dispatch(createSession())
    toast('✦', 'Nouvelle session créée')
  }

  function handleDelete(id: string) {
    dispatch(deleteSession(id))
    toast('🗑', 'Session supprimée')
  }

  return (
    <aside
      className={cn(
        'flex flex-col flex-shrink-0 bg-bg-1 border-r border-border',
        'transition-all duration-300 overflow-hidden',
        sidebarOpen ? 'w-[260px]' : 'w-0',
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2.5 px-4 py-[17px] border-b border-border flex-shrink-0">
        <div className="w-[30px] h-[30px] bg-accent rounded-sm flex items-center justify-center text-sm flex-shrink-0">⬡</div>
        <span className="font-serif text-[17px] text-tx">RAG·AI</span>
        <span className="ml-auto text-[9px] font-medium tracking-widest uppercase bg-accent/15 text-accent px-2 py-0.5 rounded-full border border-accent/25">
          RGPD
        </span>
      </div>

      {/* New session */}
      <button
        onClick={handleNew}
        className="mx-3 mt-3 mb-2 flex items-center gap-2 bg-accent text-white text-[13px] font-medium px-3.5 py-2 rounded-sm hover:bg-accent-dark active:scale-[.98] transition-all"
      >
        <Plus size={14} strokeWidth={2.5} />
        Nouvelle session
      </button>

      {/* Search */}
      <div className="relative mx-3 mb-2">
        <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-tx-3" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Rechercher…"
          className="w-full bg-bg-2 border border-border rounded-sm py-1.5 pl-7 pr-3 text-[12px] text-tx-2 placeholder-tx-3 outline-none focus:border-border-md transition-colors"
        />
      </div>

      {/* Section label */}
      <div className="px-4 pt-2 pb-1 text-[10px] font-medium tracking-widest uppercase text-tx-3 flex-shrink-0">
        Sessions récentes
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {filtered.length === 0 ? (
          <div className="text-center py-6 text-[12px] text-tx-3">Aucune session</div>
        ) : (
          filtered.map((s) => (
            <SessionItem
              key={s.id}
              session={s}
              active={s.id === currentId}
              onSelect={() => dispatch(switchSession(s.id))}
              onRename={() => dispatch(openRenameModal(s.id))}
              onDelete={() => handleDelete(s.id)}
            />
          ))
        )}
      </div>

      {/* Stats */}
      <div className="border-t border-border grid grid-cols-3 gap-2 px-3.5 py-2.5 flex-shrink-0">
        {[
          { val: sessions.length,  lbl: 'Sessions' },
          { val: totalMessages,    lbl: 'Messages' },
          { val: '—',              lbl: 'Docs RAG' },
        ].map(({ val, lbl }) => (
          <div key={lbl} className="text-center">
            <div className="font-mono text-[15px] font-medium text-tx">{val}</div>
            <div className="text-[9px] uppercase tracking-widest text-tx-3">{lbl}</div>
          </div>
        ))}
      </div>

      {/* User */}
      <div className="border-t border-border px-3.5 py-2.5 flex items-center gap-2.5 flex-shrink-0">
        <div className="w-[30px] h-[30px] rounded-full bg-gradient-to-br from-accent to-purple-500 flex items-center justify-center text-white text-[12px] font-medium flex-shrink-0">
          U
        </div>
        <div>
          <div className="text-[12.5px] font-medium text-tx">Utilisateur</div>
          <div className="text-[10px] text-tx-3">RAG RGPD · v2</div>
        </div>
        <div className="ml-auto w-[7px] h-[7px] rounded-full bg-success" />
      </div>
    </aside>
  )
}
