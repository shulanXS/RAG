'use client'

import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { Send } from 'lucide-react'
import { clsx } from 'clsx'

interface ChatInputProps {
  disabled?: boolean
  onSubmit: (content: string) => void
}

export function ChatInput({ disabled, onSubmit }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleSubmit() {
    if (!value.trim() || disabled) return
    onSubmit(value.trim())
    setValue('')

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  function handleChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setValue(e.target.value)

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }

  useEffect(() => {
    if (textareaRef.current && value === '') {
      textareaRef.current.style.height = 'auto'
    }
  }, [value])

  return (
    <div className="flex gap-2 items-end bg-white border border-border rounded-lg p-2">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Type your message..."
        rows={1}
        className={clsx(
          'flex-1 resize-none px-3 py-2 text-sm bg-transparent',
          'focus:outline-none min-h-[40px] max-h-[200px]',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
        style={{ height: 'auto' }}
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || !value.trim()}
        className={clsx(
          'shrink-0 w-10 h-10 flex items-center justify-center rounded-lg transition-colors',
          'bg-primary text-white hover:bg-primary-hover',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
      >
        <Send size={18} />
      </button>
    </div>
  )
}
