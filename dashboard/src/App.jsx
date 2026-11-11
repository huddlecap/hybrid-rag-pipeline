import React, { useState, useEffect, useRef } from 'react'

const API_BASE = 'http://localhost:8000'

function RetrievedChunkItem({ chunk, isPercentage, isCited }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const scoreLabel = isPercentage
    ? `${Math.round(chunk.display_score * 100)}%`
    : `Distance: ${Number(chunk.display_score).toFixed(4)}`
  return (
    <div id={`citation-${chunk.rank}`} className="p-3 border-b border-gray-200">
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex justify-between text-xs text-gray-500 mb-1 cursor-pointer"
      >
        <span>
          #{chunk.rank} {isExpanded ? '(Collapse)' : '(Expand)'}
        </span>
        <span>
          {chunk.source_file || 'unknown'}
          {' • '}
          {scoreLabel}
          {isCited && (
            <span className="ml-2 bg-green-100 text-green-800 text-xs font-semibold px-2 py-0.5 rounded">Cited</span>
          )}
        </span>
      </div>
      <p
        className={`text-sm text-gray-700 ${
          isExpanded ? '' : 'line-clamp-3 text-ellipsis overflow-hidden'
        }`}
      >
        {chunk.text}
      </p>
    </div>
  )
}

const GooeyLoader = () => (
  <div className="gooey-loader">
    <div className="gooey-item"></div>
    <div className="gooey-item"></div>
    <div className="gooey-item"></div>
    <div className="gooey-item"></div>
    <div className="gooey-item"></div>
  </div>
);

function App() {
  const [documents, setDocuments] = useState([])
  const [serverStatus, setServerStatus] = useState('checking')
  const [mode, setMode] = useState('standard') // 'standard' | 'compare'
  const [question, setQuestion] = useState('')
  const [leftResult, setLeftResult] = useState(null)
  const [rightResult, setRightResult] = useState(null)
  const [lastAskedQuestion, setLastAskedQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [rightLoading, setRightLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedDocs, setExpandedDocs] = useState(new Set())
  const [elapsedTime, setElapsedTime] = useState(0)
  const timerRef = useRef(null)

  useEffect(() => {
    fetchDocuments()
    checkHealth()
  }, [])

  useEffect(() => {
    return () => clearInterval(timerRef.current)
  }, [])

  async function fetchDocuments() {
    try {
      const res = await fetch(`${API_BASE}/v1/documents`)
      const data = await res.json()
      setDocuments(Array.isArray(data) ? data : [])
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

  async function ask(retrievalMode, questionText) {
    const res = await fetch(`${API_BASE}/v1/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionText, retrieval_mode: retrievalMode }),
    })
    const data = await res.json().catch((parseError) => {
      console.error(`${retrievalMode.toUpperCase()} JSON PARSE ERROR:`, parseError)
      return {}
    })
    if (retrievalMode === 'dense') {
      console.log('DENSE RAW RESPONSE:', data)
    }
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
    return data
  }

  async function handleAsk() {
    const currentQuestion = question.trim()
    if (!currentQuestion) return

    setElapsedTime(0)
    const start = Date.now()
    timerRef.current = setInterval(() => setElapsedTime(Math.floor((Date.now() - start) / 1000)), 1000)

    const isCompareMode = mode === 'compare'
    const canFetchDenseOnly =
      isCompareMode &&
      leftResult !== null &&
      currentQuestion === lastAskedQuestion

    if (canFetchDenseOnly) {
      console.log('SMART CACHE: reusing hybrid result; fetching dense only')
      setRightLoading(true)
      setError(null)
      try {
        const dense = await ask('dense', currentQuestion)
        setRightResult(dense)
        setLastAskedQuestion(currentQuestion)
      } catch (e) {
        console.error('DENSE FETCH ERROR:', e)
        setError(`Dense: ${e.message}`)
      } finally {
        clearInterval(timerRef.current)
        setRightLoading(false)
      }
      return
    }

    setLoading(true)
    setError(null)
    setLeftResult(null)
    setRightResult(null)

    const settle = (promise, label) =>
      promise.then(
        (data) => ({ ok: true, data }),
        (err) => {
          console.error(`${label} FETCH ERROR:`, err)
          return { ok: false, error: err.message || String(err) }
        }
      )

    try {
      if (isCompareMode) {
        const [left, right] = await Promise.all([
          settle(ask('hybrid', currentQuestion), 'HYBRID'),
          settle(ask('dense', currentQuestion), 'DENSE'),
        ])

        console.log('COMPARE RESULTS:', { left, right })

        if (left.ok) {
          setLeftResult(left.data)
          setLastAskedQuestion(currentQuestion)
        }
        if (right.ok) setRightResult(right.data)

        const failures = []
        if (!left.ok) failures.push(`Hybrid: ${left.error}`)
        if (!right.ok) failures.push(`Dense: ${right.error}`)
        if (failures.length) setError(failures.join(' | '))
      } else {
        const hybrid = await ask('hybrid', currentQuestion)
        setLeftResult(hybrid)
        setRightResult(null)
        setLastAskedQuestion(currentQuestion)
      }
    } catch (e) {
      console.error('ASK ERROR:', e)
      setError(e.message)
    } finally {
      clearInterval(timerRef.current)
      setLoading(false)
    }
  }

  function formatConfidence(score) {
    if (score == null) return 'N/A'
    return Math.round(score * 100) + '%'
  }

  function formatChunkFound(score) {
    if (score == null) return '0%'
    return Math.round(score * 100) + '%'
  }

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60)
    const s = secs % 60
    return m > 0 ? `${m}m ${s}s` : `${s}s`
  }

  function renderAnswerWithCitations(answer, citations) {
    if (!answer) return '(no answer)'
    const known = new Set((citations || []).map((c) => c.index))
    const parts = answer.split(/(\[\d+\])/g)
    return parts.map((part, i) => {
      const match = part.match(/^\[(\d+)\]$/)
      if (!match) return <span key={i}>{part}</span>
      const n = Number(match[1])
      if (!known.has(n)) return <span key={i} className="text-gray-400">{part}</span>
      return (
        <a
          key={i}
          href={`#citation-${n}`}
          onClick={(e) => {
            e.preventDefault()
            const el = document.getElementById(`citation-${n}`)
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
          }}
          className="text-blue-600 hover:underline font-medium"
        >
          {part}
        </a>
      )
    })
  }

  function renderAllChunks(chunks, citations) {
    if (!chunks || chunks.length === 0) return null
    return (
      <div className="mt-4 border-t pt-4">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">Top Retrieved Chunks</h3>
        {chunks.map((chunk, i) => {
          const cited = citations?.some((c) => c.text === chunk.text)
          return (
            <RetrievedChunkItem
              key={chunk.chunk_id || i}
              chunk={chunk}
              isPercentage={chunk.is_percentage}
              isCited={cited}
            />
          )
        })}
      </div>
    )
  }

  function renderResult(result, label) {
    if (!result) return null
    return (
      <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
        <div className="flex justify-between items-start mb-3">
          <h3 className="text-sm font-bold text-gray-900">{label}</h3>
          <span className="text-xs font-bold bg-green-100 text-green-700 px-2 py-0.5 rounded">
            {result.model || 'groq'}
          </span>
        </div>
        <div className="prose prose-sm max-w-none text-gray-800 mb-4 whitespace-pre-wrap">
          {renderAnswerWithCitations(result.answer, result.citations)}
        </div>
        <div className="flex gap-3 mb-3">
          <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
            Answer Relevance: {formatConfidence(result.confidence_scores?.answer_relevance)}
          </span>
          <span className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded">
            Faithfulness: {formatConfidence(result.confidence_scores?.faithfulness)}
          </span>
          <span className="text-xs px-2 py-1 bg-purple-100 text-purple-800 rounded">
            Retrieval Relevance: {formatChunkFound(result.confidence_scores?.retrieval_relevance)}
          </span>
        </div>
        {renderAllChunks(result.all_retrieved_chunks, result.citations)}
      </div>
    )
  }

  function renderLeftColumn() {
    if (!leftResult) {
      return (
        <div className="h-48 flex items-center justify-center bg-gray-50 rounded border border-gray-200">
          <span className="text-gray-500 text-sm">Ask a question to see hybrid results</span>
        </div>
      )
    }
    return <div className="space-y-4">{renderResult(leftResult, 'Hybrid')}</div>
  }

  function renderRightColumn() {
    if (rightLoading) {
      return (
        <div className="h-48 flex items-center justify-center bg-gray-50 rounded border border-gray-200">
          <span className="text-gray-500 text-sm">Fetching dense comparison data...</span>
        </div>
      )
    }
    if (!rightResult) {
      const message = leftResult
        ? 'Click Ask to fetch dense comparison data.'
        : 'Ask a question to see dense results'
      return (
        <div className="h-48 flex items-center justify-center bg-gray-50 rounded border border-gray-200">
          <span className="text-gray-500 text-sm">{message}</span>
        </div>
      )
    }
    return <div className="space-y-4">{renderResult(rightResult, 'Dense')}</div>
  }

  function toggleDoc(sourceFile) {
    setExpandedDocs((prev) => {
      const next = new Set(prev)
      if (next.has(sourceFile)) next.delete(sourceFile)
      else next.add(sourceFile)
      return next
    })
  }

  const isExpanded = (sourceFile) => expandedDocs.has(sourceFile)

  const totalChunks = documents.reduce((sum, file) => sum + (file.total_chunks || 0), 0)

  console.log('RIGHT STATE:', rightResult)

  return (
    <div className="w-full h-screen bg-gray-50 flex flex-col overflow-hidden">
      <svg xmlns="http://www.w3.org/2000/svg" version="1.1" className="hidden">
        <defs>
          <filter id="goo">
            <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
            <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" result="goo" />
            <feBlend in="SourceGraphic" in2="goo" />
          </filter>
        </defs>
      </svg>
      {/* Header */}
      <header className="shrink-0 bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900 m-0 leading-tight">RAG Evaluation</h1>
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

      <div className="flex flex-col md:flex-row flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-white border-r border-gray-200 p-6 overflow-y-auto h-full">
          <h2 className="text-sm font-bold text-gray-900 mb-4">Documents ({totalChunks} chunks)</h2>
          {documents.length === 0 ? (
            <p className="text-xs text-gray-500">No documents indexed</p>
          ) : (
            <div className="space-y-1">
              {documents.map((file) => {
                const isExpanded = expandedDocs.has(file.source_file)
                return (
                  <div key={file.source_file} className="border border-gray-200 rounded-lg overflow-hidden">
                    <div
                      onClick={() => toggleDoc(file.source_file)}
                      className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
                    >
                      <span className="flex items-center gap-2 text-sm font-medium text-gray-700 min-w-0">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4 text-gray-400 shrink-0">
                          <path fillRule="evenodd" d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0 0 16.5 9h-1.875a1.875 1.875 0 0 1-1.875-1.875V5.25A3.75 3.75 0 0 0 9 1.5H5.625ZM7.5 15a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5A.75.75 0 0 1 7.5 15Zm.75 2.25a.75.75 0 0 0 0 1.5H12a.75.75 0 0 0 0-1.5H8.25Z" clipRule="evenodd" />
                          <path d="M12.971 1.816A5.23 5.23 0 0 1 14.25 5.25v1.875c0 .207.168.375.375.375H16.5a5.23 5.23 0 0 1 3.434 1.279 9.768 9.768 0 0 0-6.963-6.963Z" />
                        </svg>
                        <span className="flex-1 min-w-0 truncate" title={file.source_file}>{file.source_file}</span>
                        <span className="shrink-0 text-xs bg-gray-200 text-gray-600 px-1.5 py-0.5 rounded">
                          {file.total_chunks} chunk{file.total_chunks > 1 ? 's' : ''}
                        </span>
                      </span>
                      <span className={`shrink-0 text-xs ${isExpanded ? 'text-green-600' : 'text-gray-400'}`}>
                        {isExpanded ? '▾' : '▸'}
                      </span>
                    </div>
                    {isExpanded && (
                      <div className="pl-4 space-y-1">
                        {file.chunks.map((chunk, i) => (
                          <div key={chunk.chunk_id || i} className="px-3 py-1.5 text-xs bg-white hover:bg-gray-50 rounded-md transition-colors">
                            <span className="text-gray-500">chunk #{chunk.chunk_index}</span>
                            <p className="text-gray-600 whitespace-pre-wrap break-words mt-0.5">{chunk.text}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto h-full p-6">
          {/* Mode Toggle */}
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-lg font-bold text-gray-900">RAG Evaluation</h2>
              <p className="text-sm text-gray-600">Compare retrieval strategies side by side</p>
            </div>
            <div className="flex bg-gray-200 p-1 rounded-lg w-fit">
              <button
                onClick={() => setMode('standard')}
                className={
                  mode === 'standard'
                    ? 'bg-white shadow-sm text-gray-900 rounded-md px-4 py-1.5 text-sm font-medium transition-all'
                    : 'text-gray-500 hover:text-gray-700 px-4 py-1.5 text-sm font-medium transition-all'
                }
              >
                Standard Mode
              </button>
              <button
                onClick={() => setMode('compare')}
                className={
                  mode === 'compare'
                    ? 'bg-white shadow-sm text-gray-900 rounded-md px-4 py-1.5 text-sm font-medium transition-all'
                    : 'text-gray-500 hover:text-gray-700 px-4 py-1.5 text-sm font-medium transition-all'
                }
              >
                Compare Mode
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="p-4 rounded bg-red-50 border border-red-200 text-red-700 text-sm mb-6">
              {error}
            </div>
          )}

          {/* Question form, shared by both modes */}
          <div className="space-y-4 mb-6">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              className="w-full bg-white text-gray-900 border border-gray-200 rounded-md p-4 resize-none shadow-[0_8px_30px_rgb(0,0,0,0.04)] transition-all duration-300 outline-none focus:outline-none focus:border-gray-400 focus:ring-1 focus:ring-gray-400 focus:shadow-[0_10px_40px_rgb(0,0,0,0.08)]"
              placeholder="Ask a question about your documents..."
            />
            <div className="flex items-center">
              {loading || rightLoading ? (
                <GooeyLoader />
              ) : (
                <button
                  onClick={handleAsk}
                  disabled={!question.trim()}
                  className="bg-black text-white px-6 py-2 rounded font-semibold disabled:opacity-50 hover:bg-gray-800"
                >
                  Ask Question
                </button>
              )}
              {elapsedTime > 0 && <span className="ml-3 font-normal text-gray-400 tabular-nums text-sm">{formatTime(elapsedTime)}</span>}
            </div>
          </div>

          {/* Standard Mode */}
          {mode === 'standard' && leftResult && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              {renderResult(leftResult, 'Standard Answer')}
            </div>
          )}

          {/* Compare Mode */}
          {mode === 'compare' && (
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="border-l-2 border-blue-500 pl-4 mb-4">
                  <h3 className="text-sm font-bold text-blue-700">Hybrid Retrieval</h3>
                </div>
                {renderLeftColumn()}
              </div>
              <div className="space-y-4">
                <div className="border-l-2 border-green-500 pl-4 mb-4">
                  <h3 className="text-sm font-bold text-green-700">Dense Retrieval</h3>
                </div>
                {renderRightColumn()}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App