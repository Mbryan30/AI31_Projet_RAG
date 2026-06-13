import { useState } from 'react'
import { Copy, RotateCcw, ChevronDown, ChevronUp } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn, formatTime } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import { useToast } from '@/hooks/useToast'
import type { Message } from '@/types'

interface Props {
  message:      Message
  onRegenerate?: () => void
}

export function MessageBubble({ message, onRegenerate }: Props) {
  const { toast } = useToast()
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const isUser = message.role === 'user'

  function handleCopy() {
    navigator.clipboard.writeText(message.content)
    toast('✓', 'Copié dans le presse-papier')
  }

  return (
    <div className={cn('group flex gap-3 mb-6 animate-fadeUp', isUser && 'flex-row')}>
      {/* Avatar */}
      <div className={cn(
        'w-7 h-7 rounded-sm flex items-center justify-center text-xs flex-shrink-0 mt-0.5',
        isUser
          ? 'bg-gradient-to-br from-accent to-purple-500 text-white font-medium'
          : 'bg-bg-2 border border-border-md text-tx-2',
      )}>
        {isUser ? 'U' : '⬡'}
      </div>

      <div className="flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-baseline gap-2 mb-1.5">
          <span className="text-[12px] font-medium text-tx">
            {isUser ? 'Vous' : 'RAG·AI'}
          </span>
          {!isUser && (
            <span className="inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-success/10 border border-success/20 text-success">
              RAG
            </span>
          )}
          <span className="text-[10px] text-tx-3">{formatTime(message.ts)}</span>
        </div>

        {/* Content */}
        <div className="text-[14px] text-tx leading-relaxed prose prose-invert prose-sm max-w-none">
          {message.isStreaming ? (
            <span>
              {message.content}
              <span className="inline-block w-0.5 h-4 bg-accent ml-0.5 animate-blink align-middle" />
            </span>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ className, children }) {
                  const isBlock = className?.includes('language-')
                  return isBlock ? (
                    <pre className="bg-bg-2 border border-border rounded-sm p-3.5 overflow-x-auto my-3">
                      <code className="font-mono text-[12.5px] text-[#a8d8ff] leading-relaxed">{children}</code>
                    </pre>
                  ) : (
                    <code className="font-mono text-[12px] bg-bg-2 border border-border px-1.5 py-0.5 rounded text-[#a8d8ff]">
                      {children}
                    </code>
                  )
                },
                p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                strong: ({ children }) => <strong className="font-medium text-tx">{children}</strong>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Sources */}
        {!isUser && !message.isStreaming && message.sources && message.sources.length > 0 && (
          <div className="mt-2.5 bg-bg-1 border border-border rounded-sm overflow-hidden">
            <button
              onClick={() => setSourcesOpen((v) => !v)}
              className="w-full flex items-center gap-2 px-3 py-2 text-[11px] text-tx-2 hover:bg-white/[.02] transition-colors"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
              </svg>
              {message.sources.length} source{message.sources.length > 1 ? 's' : ''} récupérée{message.sources.length > 1 ? 's' : ''}
              <span className="ml-auto">{sourcesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}</span>
            </button>
            {sourcesOpen && (
              <div className="px-3 pb-3 pt-1 space-y-1.5">
                {message.sources.map((src, i) => (
                  <div key={i} className="flex items-center gap-2 bg-bg-2 border border-border rounded-[7px] px-2.5 py-1.5 hover:border-accent/40 transition-colors">
                    <span className="w-[18px] h-[18px] flex items-center justify-center rounded-[5px] bg-accent/15 text-accent font-mono text-[10px] font-medium flex-shrink-0">
                      {i + 1}
                    </span>
                    <span className="flex-1 text-[12px] text-tx-2 truncate">{src.title}</span>
                    <span className="font-mono text-[10px] text-success">{src.score}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Metrics */}
        {!isUser && !message.isStreaming && message.metrics && (
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {[
              { dot: '#3ddc84', label: `${message.metrics.latency}ms` },
              { dot: '#6c8fff', label: `${message.metrics.tokens} tokens` },
              { dot: '#f5a623', label: message.metrics.strategy },
            ].map(({ dot, label }) => (
              <div key={label} className="flex items-center gap-1 text-[10px] font-mono text-tx-3 px-2 py-0.5 rounded-full border border-border">
                <div className="w-1 h-1 rounded-full flex-shrink-0" style={{ background: dot }} />
                {label}
              </div>
            ))}
          </div>
        )}

        {/* Actions */}
        {!message.isStreaming && (
          <div className="flex gap-1.5 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button size="sm" onClick={handleCopy}>
              <Copy size={11} /> Copier
            </Button>
            {!isUser && onRegenerate && (
              <Button size="sm" onClick={onRegenerate}>
                <RotateCcw size={11} /> Regénérer
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
