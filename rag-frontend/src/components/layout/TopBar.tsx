import { Menu, Download, Trash2 } from 'lucide-react'
import { useAppDispatch, useAppSelector } from '@/hooks/redux'
import { useToast } from '@/hooks/useToast'
import { selectCurrent, clearMessages } from '@/store/slices/sessionsSlice'
import { toggleSidebar, selectSidebarOpen } from '@/store/slices/uiSlice'
import { Button } from '@/components/ui/Button'

export function TopBar() {
  const dispatch  = useAppDispatch()
  const session   = useAppSelector(selectCurrent)
  const sidebarOpen = useAppSelector(selectSidebarOpen)
  const { toast } = useToast()

  function handleExport() {
    if (!session) return
    const lines = [
      `# ${session.title}\n`,
      `Créée le ${new Date(session.created).toLocaleString('fr-FR')}\n\n`,
    ]
    session.messages.forEach((m) => {
      lines.push(`**${m.role === 'user' ? 'Vous' : 'RAG·AI'}** — ${new Date(m.ts).toLocaleTimeString('fr-FR')}`)
      lines.push(m.content + '\n')
    })
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' })
    const a = Object.assign(document.createElement('a'), {
      href:     URL.createObjectURL(blob),
      download: `session-${session.title.replace(/\s+/g, '-').toLowerCase()}.md`,
    })
    a.click()
    toast('⬇', 'Export Markdown téléchargé')
  }

  function handleClear() {
    dispatch(clearMessages())
    toast('🧹', 'Conversation vidée')
  }

  return (
    <header className="border-b border-border h-[52px] px-6 flex items-center gap-3 flex-shrink-0">
      {/* Hamburger (hidden when sidebar open) */}
      {!sidebarOpen && (
        <Button variant="icon" size="md" onClick={() => dispatch(toggleSidebar())}>
          <Menu size={14} />
        </Button>
      )}

      {/* Title */}
      <h1 className="font-serif text-[16px] text-tx flex-1 truncate">
        {session?.title ?? 'Nouvelle session'}
      </h1>

      {/* Status pill */}
      <div className="flex items-center gap-1.5 text-[11px] text-tx-2 bg-bg-2 border border-border rounded-full px-3 py-1">
        <div className="w-1.5 h-1.5 rounded-full bg-success" />
        Pipeline actif
      </div>

      {/* Actions */}
      <Button size="sm" onClick={handleExport}>
        <Download size={12} /> Exporter
      </Button>
      <Button variant="icon" size="md" onClick={handleClear} title="Vider la conversation">
        <Trash2 size={13} />
      </Button>
    </header>
  )
}
