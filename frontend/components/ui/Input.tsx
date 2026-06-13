import { clsx } from 'clsx'
import { forwardRef } from 'react'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={clsx(
          'flex h-10 w-full rounded-lg border border-border bg-white px-3 py-2 text-sm',
          'placeholder:text-foreground-secondary',
          'focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent',
          'disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        {...props}
      />
    )
  }
)

Input.displayName = 'Input'

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={clsx(
          'flex min-h-[80px] w-full rounded-lg border border-border bg-white px-3 py-2 text-sm',
          'placeholder:text-foreground-secondary',
          'focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent',
          'disabled:cursor-not-allowed disabled:opacity-50',
          'resize-none',
          className
        )}
        {...props}
      />
    )
  }
)

Textarea.displayName = 'Textarea'
