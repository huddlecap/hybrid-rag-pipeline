import React, { useState, useEffect } from 'react'

const API_BASE = 'http://localhost:8000'

function App() {
  const [documents, setDocuments] = useState([])
  const [serverStatus, setServerStatus] = useState('checking')
  const [mode, setMode] = useState('standard') // 'standard' | 'compare'
  const [question, setQuestion] = useState('')
  const [leftResult, setLeftResult] = useState(null)
  const [rightResult, setRightResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchDocuments()
    checkHealth()
  }, [])

  async function fetchDocuments() {
    try {
      const res = await fetch(`${API_BASE}/v1/documents`)
      const data = await res.json()
      setDocuments(data.documents || [])
    } catch (e) {
      console.warn('Could not fetch documents:', e.message)
    }
  }

  async function checkHealth() {
    try {
      await fetch(`${API_BASE}/`)
      setServerStatus('online')
    } catch (e) {
      setServerStatus('offline')
    }
  }

  async function ask(mode, question) {
    const res = await fetch(`${API_BASE}/v1/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, retrieval_mode: mode }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
  }

  async function handleAsk() {
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    try {
      if (mode === 'compare') {
        const [left, right] = await Promise.all([
          ask('hybrid', question),
          ask('dense', question),
        ])
        setLeftResult(left)
        setRightResult(right)
      } else {
        const res = await ask('hybrid', question)
        setLeftResult(res)
        setRightResult(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function formatConfidence(score) {
    if (score == null) return 'N/A'
    return Math.round(score * 100)
  }

  function renderChunkList(results) {
    if (!results || results.length === 0) return null
    return (
      <div className="mt-4 border-t pt-4">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">Retrieved Chunks</h3>
        {results.map((chunk, i) => (
          <div key={chunk.chunk_id || i} className="p-3 border-b border-gray-200 bg-gray-50">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>#{i + 1}</span>
              <span>{chunk.metadata?.source_file || 'unknown'}</span>
            </div>
            <p className="text-sm text-gray-700">
              {chunk.text.slice(0, 200)}{chunk.text.length > 200 ? '…' : ''}
            </p>
          </div>
        ))}
      </div>
    )
  }

  function renderResult(result, label) {
    if (!result) return null
    return (
      <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
        <div className="flex justify-between items-start mb-3">
          <h3 className="text-sm font-semibold text-gray-900">{label}</h3>
          <span className="text-xs font-bold bg-green-100 text-green-700 px-2 py-0.5 rounded">
            {result.model || 'groq'}
          </span>
        </div>
        <div className="prose prose-sm max-w-none text-gray-800 mb-4 whitespace-pre-wrap">
          {result.answer || '(no answer)'}
        </div>
        <div className="flex gap-3 mb-3">
          <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
            Correctness: {formatConfidence(result.confidence_scores?.correctness)}%
          </span>
          <span className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded">
            Faithfulness: {formatConfidence(result.confidence_scores?.faithfulness)}%
          </span>
          <span className="text-xs px-2 py-1 bg-purple-100 text-purple-800 rounded">
            Retrieval: {formatConfidence(result.confidence_scores?.retrieval_relevance)}%
          </span>
        </div>
        {renderChunkList(result.results)}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900">RAG Evaluation</h1>
          <div className="flex items-center gap-2">
            <span className={`text-xs font-bold px-2 py-1 rounded ${
              serverStatus === 'online' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {serverStatus}
            </span>
            <span className="text-xs text-gray-500">Port 8000</span>
          </div>
        </div>
      </header>

      <div className="flex flex-col md:flex-row h-[calc(100vh-56px)]">
        {/* Sidebar */}
        <aside className="w-80 bg-white border-r border-gray-200 p-6">
          <h2 className="text-sm font-bold text-gray-900 mb-4">Documents</h2>
          {documents.length === 0 ? (
            <p className="text-xs text-gray-500">No documents indexed</p>
          ) : (
            <ul className="space-y-2">
              {documents.map((doc, i) => (
                <div key={i} className="flex items-center gap-2 text-xs bg-gray-50 px-3 py-2 rounded">
                  <span className="text-gray-400">📄</span>
                  <span className="truncate">{doc.source_file || doc.chunk_id}</span>
                </div>
              ))}
            </ul>
          )}
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-6">
          {/* Mode Toggle */}
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-lg font-bold text-gray-900">RAG Evaluation</h2>
              <p className="text-sm text-gray-600">Compare retrieval strategies side by side</p>
            </div>
            <label className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
              <span className="text-xs font-semibold text-gray-600">Standard Mode</span>
              <input
                type="checkbox"
                checked={mode === 'compare'}
                onChange={(e) => setMode(e.target.checked ? 'compare' : 'standard')}
                className="w-4 h-4 rounded bg-blue-600"
              />
              <span className="text-xs font-semibold text-gray-600">Compare Mode</span>
            </label>
          </div>

          {/* Error */}
          {error && (
            <div className="p-4 rounded bg-red-50 border border-red-200 text-red-700 text-sm mb-6">
              {error}
            </div>
          )}

          {/* Standard Mode */}
          {mode === 'standard' && (
            <div className="space-y-4">
              <div className="border rounded-lg p-4">
                <label className="text-xs font-semibold text-gray-700 mb-2 block">Question</label>
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={3}
                  className="w-full border border-gray-300 rounded p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                  placeholder="Ask a question about your documents..."
                />
              </div>
              <button
                onClick={handleAsk}
                disabled={loading || !question.trim()}
                className="bg-blue-600 text-white px-6 py-2 rounded font-semibold disabled:opacity-50 hover:bg-blue-700"
              >
                {loading ? 'Asking...' : 'Ask Question'}
              </button>
              {leftResult && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                  {renderResult(leftResult, 'Standard Answer')}
                </div>
              )}
            </div>
          )}

          {/* Compare Mode */}
          {mode === 'compare' && (
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="border-l-2 border-blue-500 pl-4 mb-4">
                  <h3 className="text-sm font-bold text-blue-700">Hybrid Retrieval</h3>
                </div>
                {leftResult ? (
                  <div className="space-y-4">
                    {renderResult(leftResult, 'Hybrid')}
                  </div>
                ) : (
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded border border-gray-200">
                    <span className="text-gray-500 text-sm">Ask a question to see hybrid results</span>
                  </div>
                )}
              </div>
              <div className="space-y-4">
                <div className="border-l-2 border-green-500 pl-4 mb-4">
                  <h3 className="text-sm font-bold text-green-700">Dense Retrieval</h3>
                </div>
                {rightResult ? (
                  <div className="space-y-4">
                    {renderResult(rightResult, 'Dense')}
                  </div>
                ) : (
                  <div className="h-48 flex items-center justify-center bg-gray-50 rounded border border-gray-200">
                    <span className="text-gray-500 text-sm">Ask a question to see dense results</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App