'use client'

import { useChat } from '@/hooks/useChat'
import { ChatMessage } from '@/components/chat/ChatMessage'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/Button'

export default function ChatPage() {
  const { messages, isLoading, sendMessage, clearMessages } = useChat()

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Chat</h2>
        <Button variant="secondary" size="sm" onClick={clearMessages}>
          Clear
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col gap-4 mb-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-foreground-secondary">
            <p className="text-sm">Start a conversation by typing a message below.</p>
          </div>
        )}
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
      </div>

      <div className="shrink-0">
        <ChatInput disabled={isLoading} onSubmit={sendMessage} />
      </div>
    </div>
  )
}
