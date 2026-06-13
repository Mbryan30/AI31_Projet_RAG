import { useEffect, useRef } from 'react'
import { useAppSelector } from '@/hooks/redux'
import { selectCurrent } from '@/store/slices/sessionsSlice'
import { formatDate } from '@/lib/utils'
import { MessageBubble } from './MessageBubble'
import { WelcomeScreen } from './WelcomeScreen'

export function ChatArea() {
  const session    = useAppSelector(selectCurrent)
  const bottomRef  = useRef<HTMLDivElement>(null)
  const messages   = session?.messages ?? []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, messages[messages.length - 1]?.content])

  if (!session || messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto">
        <WelcomeScreen />
      </div>
    )
  }

  // Group by date for dividers
  const groups: { date: string; indices: number[] }[] = []
  messages.forEach((m, i) => {
    const d = formatDate(m.ts)
    const last = groups[groups.length - 1]
    if (last && last.date === d) { last.indices.push(i) }
    else { groups.push({ date: d, indices: [i] }) }
  })

  return (
    <div className="flex-1 overflow-y-auto py-7">
      <div className="max-w-[760px] mx-auto px-6">
        {groups.map((group) => (
          <div key={group.date}>
            {/* Date divider */}
            <div className="flex items-center gap-3 my-5">
              <div className="flex-1 h-px bg-border" />
              <span className="text-[10px] uppercase tracking-widest text-tx-3">{group.date}</span>
              <div className="flex-1 h-px bg-border" />
            </div>
            {group.indices.map((i) => (
              <MessageBubble key={messages[i].id} message={messages[i]} />
            ))}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
