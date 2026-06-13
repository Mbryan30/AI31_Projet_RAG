import { Pencil, Trash2 } from 'lucide-react'
import { cn, timeAgo } from '@/lib/utils'
import type { Session } from '@/types'

interface Props {
  session:  Session
  active:   boolean
  onSelect: () => void
  onRename: () => void
  onDelete: () => void
}

export function SessionItem({ session, active, onSelect, onRename, onDelete }: Props) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => e.key === 'Enter' && onSelect()}
      className={cn(
        'group flex items-center gap-2 px-2.5 py-2 rounded-[9px] cursor-pointer',
        'transition-colors duration-100 select-none',
        active ? 'bg-bg-3' : 'hover:bg-bg-2',
      )}
    >
      {/* dot */}
      <span className={cn(
        'w-1.5 h-1.5 rounded-full flex-shrink-0 transition-colors',
        active ? 'bg-accent' : 'bg-tx-3',
      )} />

      {/* title */}
      <span className={cn(
        'flex-1 text-[12.5px] truncate transition-colors',
        active ? 'text-tx' : 'text-tx-2',
      )}>
        {session.title}
      </span>

      {/* time — hidden when hovering */}
      <span className="text-[10px] text-tx-3 flex-shrink-0 group-hover:hidden">
        {timeAgo(session.updated)}
      </span>

      {/* action buttons — visible on hover */}
      <div className="hidden group-hover:flex items-center gap-0.5">
        <button
          onClick={(e) => { e.stopPropagation(); onRename() }}
          className="p-1 rounded text-tx-3 hover:text-tx hover:bg-white/[.06] transition-colors"
          title="Renommer"
        >
          <Pencil size={11} />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          className="p-1 rounded text-tx-3 hover:text-danger hover:bg-white/[.06] transition-colors"
          title="Supprimer"
        >
          <Trash2 size={11} />
        </button>
      </div>
    </div>
  )
}
