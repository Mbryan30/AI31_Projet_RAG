import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'ghost' | 'icon'
  size?:    'sm' | 'md'
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'ghost', size = 'md', className, children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center gap-1.5 rounded-sm font-sans transition-all focus-visible:outline-none',
          // variants
          variant === 'primary' && 'bg-accent text-white hover:bg-accent-dark active:scale-[.97]',
          variant === 'ghost'   && 'border border-border text-tx-2 hover:border-border-md hover:bg-bg-2 hover:text-tx',
          variant === 'icon'    && 'border border-border text-tx-3 hover:bg-bg-2 hover:text-tx',
          // sizes
          size === 'md' && 'px-3 py-1.5 text-[13px]',
          size === 'sm' && 'px-2 py-1 text-[11px]',
          variant === 'icon' && size === 'md' && 'h-8 w-8 p-0',
          variant === 'icon' && size === 'sm' && 'h-6 w-6 p-0',
          props.disabled && 'opacity-40 cursor-not-allowed',
          className,
        )}
        {...props}
      >
        {children}
      </button>
    )
  },
)
Button.displayName = 'Button'
