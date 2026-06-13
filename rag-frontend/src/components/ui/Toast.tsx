import { useAppSelector } from '@/hooks/redux'
import { selectToast } from '@/store/slices/uiSlice'
import { cn } from '@/lib/utils'

export function Toast() {
  const toast = useAppSelector(selectToast)

  return (
    <div
      className={cn(
        'fixed bottom-6 right-6 z-50 flex items-center gap-2.5',
        'bg-bg-2 border border-border-md rounded-[10px] px-4 py-2.5',
        'text-[13px] text-tx shadow-lg',
        'transition-all duration-200',
        toast.visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none',
      )}
    >
      <span className="text-sm">{toast.icon}</span>
      <span>{toast.message}</span>
    </div>
  )
}
