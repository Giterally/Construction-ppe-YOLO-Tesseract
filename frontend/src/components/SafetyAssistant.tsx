import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { API_URL } from '../config'

interface Message {
  id: string
  user_message: string
  ai_response: string
  created_at: string
  analysis_id?: string
}


export default function SafetyAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadChatHistory()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadChatHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/api/chat/history`)
      if (response.ok) {
        const data = await response.json()
        setMessages(data.messages || [])
      }
    } catch (err) {
      console.error('Failed to load chat history:', err)
    }
  }


  const sendMessage = async (message: string) => {
    if (!message.trim()) return

    setLoading(true)
    setError(null)
    setInputMessage('')

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to get response')
      }

      const data = await response.json()

      // Add message to list
      const newMessage: Message = {
        id: data.message_id || `msg-${Date.now()}`,
        user_message: message,
        ai_response: data.response,
        created_at: new Date().toISOString(),
        analysis_id: data.analysis_id
      }

      setMessages([...messages, newMessage])

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message')
    } finally {
      setLoading(false)
    }
  }

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(inputMessage)
  }

  const handleSendQuestion = (question: string) => {
    sendMessage(question)
  }

  const handleClearHistory = async () => {
    if (!window.confirm('Clear all chat history?')) return

    try {
      const response = await fetch(`${API_URL}/api/chat/clear`, {
        method: 'POST',
      })

      if (response.ok) {
        setMessages([])
      }
    } catch (err) {
      setError('Failed to clear chat history')
    }
  }

  return (
    <div className="assistant-container">
      {/* Info Banner */}
      <div style={{ 
        padding: '12px 16px', 
        background: '#f8f9fa', 
        border: '1px solid var(--color-border)',
        marginBottom: '16px',
        fontSize: '14px',
        color: '#666',
        textAlign: 'center'
      }}>
        💬 This chat assistant has access to <strong>all your past photo analyses</strong> and can answer questions about trends, compliance history, and site safety across all uploaded images.
      </div>

      {/* Chat Messages */}
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="empty-chat">
            <h3>👋 Ask me anything about construction safety</h3>
            <p style={{ fontSize: '14px', color: '#666', marginBottom: '20px' }}>
              I have access to <strong>all your past photo analyses</strong> and can answer questions about trends, compliance history, and site safety across all your uploaded images.
            </p>
            <div className="example-questions">
              <p><strong>Try asking:</strong></p>
              <ul>
                <li onClick={() => handleSendQuestion("What's the compliance trend across all my analyses?")}>
                  "What's the compliance trend across all my analyses?"
                </li>
                <li onClick={() => handleSendQuestion("Compare the latest photo to previous ones")}>
                  "Compare the latest photo to previous ones"
                </li>
                <li onClick={() => handleSendQuestion("What PPE is required based on all site photos?")}>
                  "What PPE is required based on all site photos?"
                </li>
                <li onClick={() => handleSendQuestion("Is this scaffold compliant with CDM 2015?")}>
                  "Is this scaffold compliant with CDM 2015?"
                </li>
                <li onClick={() => handleSendQuestion("How many workers were detected across all analyses?")}>
                  "How many workers were detected across all analyses?"
                </li>
              </ul>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className="message-group">
            <div className="message user-message">
              <div className="message-sender">You</div>
              <div className="message-content">{msg.user_message}</div>
            </div>
            <div className="message ai-message">
              <div className="message-sender">Safety Assistant</div>
              <div className="message-content">
                <ReactMarkdown>{msg.ai_response}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="message ai-message">
            <div className="message-sender">Safety Assistant</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            {error}
            <button onClick={() => setError(null)}>✕</button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={handleSendMessage} className="chat-input-form">
        <div className="chat-input-wrapper">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask about safety compliance..."
            disabled={loading}
            className="chat-input"
          />
          <button 
            type="submit" 
            disabled={loading || !inputMessage.trim()}
            className="btn-send"
          >
            {loading ? '...' : '→'}
          </button>
        </div>
        <div className="chat-actions">
          <button 
            type="button"
            onClick={handleClearHistory}
            className="btn-clear-chat"
          >
            Clear History
          </button>
        </div>
      </form>

    </div>
  )
}

