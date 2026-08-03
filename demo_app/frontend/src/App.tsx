import { useState, useEffect, useRef } from 'react'
import StageIndicator from './components/StageIndicator'
import DebateTranscript from './components/DebateTranscript'
import VerdictCard from './components/VerdictCard'
import EvidencePanel from './components/EvidencePanel'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Verdict {
  label: string
  verdict_binary: 'REAL' | 'FAKE'
  verdict_label_vi: 'Thật' | 'Giả'
  confidence: number
  rationale: string
  citations: string[]
  recommendation: string
}

export interface ArgumentScore {
  agent: string
  round: number
  factuality: number
  rebuttal_engagement: number
  evidence_grounding: number
}

export interface WeightBreakdown {
  phobert: number
  coolant: number
  evidence: number
  argument_scores: ArgumentScore[]
}

export interface Evidence {
  title: string
  url: string
  snippet: string
  source_tier: 'trusted' | 'flagged' | 'social' | 'unknown'
}

export interface DebateTurn {
  agent: 'real_advocate' | 'fake_advocate'
  round: number
  text: string
  timestamp: string
  error?: string
}

// ── Stage constants (D-10) ────────────────────────────────────────────────────

export const STAGES = ['evidence_retrieval', 'reranking', 'verification', 'debate', 'verdict'] as const
export type StageName = typeof STAGES[number]

export const STAGE_LABELS: Record<StageName, string> = {
  evidence_retrieval: 'Tìm bằng chứng',
  reranking:          'Xếp hạng bằng chứng',
  verification:       'Kiểm định mô hình',
  debate:             'Tranh luận',
  verdict:            'Phán quyết',
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  // Form state
  const [statement, setStatement] = useState('')
  const [imageUrl, setImageUrl] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)

  // Analysis state
  const [requestId, setRequestId] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isError, setIsError] = useState(false)

  // Pipeline stage state (D-09)
  const [currentStage, setCurrentStage] = useState<StageName | null>(null)
  const [completedStages, setCompletedStages] = useState<StageName[]>([])

  // Debate transcript state
  const [allTurns, setAllTurns] = useState<DebateTurn[]>([])
  const [currentTurnAgent, setCurrentTurnAgent] = useState<string | null>(null)
  const [currentTurnRound, setCurrentTurnRound] = useState<number>(0)
  const currentTurnTextRef = useRef('')
  const [currentTurnText, setCurrentTurnText] = useState('')

  // Verdict + evidence state (revealed together at verdict event, D-05/D-07)
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [weightBreakdown, setWeightBreakdown] = useState<WeightBreakdown | null>(null)
  const [evidenceReal, setEvidenceReal] = useState<Evidence[]>([])
  const [evidenceFake, setEvidenceFake] = useState<Evidence[]>([])
  const [showEvidence, setShowEvidence] = useState(false)
  const [showBadges, setShowBadges] = useState(false)

  // SSE: StrictMode-safe EventSource lifecycle (DEMO-04)
  useEffect(() => {
    if (!requestId) return

    const es = new EventSource(`http://localhost:8000/api/stream/${requestId}`)
    setIsStreaming(true)

    es.addEventListener('stage_start', (e) => {
      const data = JSON.parse(e.data) as { type: string; name: string }
      const name = data.name as StageName
      setCurrentStage(name)
      setCompletedStages((prev) => (prev.includes(name) ? prev : [...prev, name]))
    })

    es.addEventListener('turn_start', (e) => {
      const data = JSON.parse(e.data) as { type: string; agent: string; round: number }
      setCurrentTurnAgent(data.agent)
      setCurrentTurnRound(data.round)
      currentTurnTextRef.current = ''
      setCurrentTurnText('')
    })

    es.addEventListener('chunk', (e) => {
      const data = JSON.parse(e.data) as { type: string; text: string }
      currentTurnTextRef.current += data.text
      setCurrentTurnText(currentTurnTextRef.current)
    })

    es.addEventListener('turn_end', (e) => {
      const data = JSON.parse(e.data) as { type: string; agent: string; round: number }
      const finalText = currentTurnTextRef.current
      setAllTurns((prev) => [
        ...prev,
        {
          agent: data.agent as 'real_advocate' | 'fake_advocate',
          round: data.round,
          text: finalText,
          timestamp: new Date().toISOString(),
        },
      ])
      setCurrentTurnAgent(null)
      setCurrentTurnRound(0)
      currentTurnTextRef.current = ''
      setCurrentTurnText('')
    })

    es.addEventListener('verdict', (e) => {
      const data = JSON.parse(e.data) as {
        type: string
        verdict: Verdict
        weight_breakdown: WeightBreakdown
        evidence_real: Evidence[]
        evidence_fake: Evidence[]
        debate_turns: DebateTurn[]
      }
      setVerdict(data.verdict)
      setWeightBreakdown(data.weight_breakdown)
      setEvidenceReal(data.evidence_real ?? [])
      setEvidenceFake(data.evidence_fake ?? [])
      setShowEvidence(true)   // D-05
      setShowBadges(true)     // D-07
      setIsStreaming(false)
      es.close()
    })

    es.onerror = () => {
      setIsError(true)
      setIsStreaming(false)
      es.close()
    }

    return () => es.close()   // StrictMode cleanup (DEMO-04)
  }, [requestId])

  const handleSubmit = async () => {
    if (!statement.trim()) return
    // Reset state for new analysis
    setIsError(false)
    setCurrentStage(null)
    setCompletedStages([])
    setAllTurns([])
    setCurrentTurnAgent(null)
    setCurrentTurnRound(0)
    currentTurnTextRef.current = ''
    setCurrentTurnText('')
    setVerdict(null)
    setWeightBreakdown(null)
    setEvidenceReal([])
    setEvidenceFake([])
    setShowEvidence(false)
    setShowBadges(false)

    const fd = new FormData()
    fd.append('statement', statement)
    if (imageUrl.trim()) fd.append('image_url', imageUrl.trim())
    if (imageFile) fd.append('image_file', imageFile)

    try {
      const res = await fetch('http://localhost:8000/api/analyze', { method: 'POST', body: fd })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json() as { request_id: string }
      setRequestId(json.request_id)
    } catch {
      setIsError(true)
    }
  }

  const handleRetry = () => {
    setIsError(false)
    setRequestId(null)
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Kiểm tra tin giả</h1>

      {/* Submission form */}
      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Nhập câu cần kiểm tra
        </label>
        <textarea
          className="w-full border border-gray-300 rounded-lg p-3 mb-4 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={3}
          placeholder="Ví dụ: Vắc-xin COVID-19 gây ra bệnh tự kỷ..."
          value={statement}
          onChange={(e) => setStatement(e.target.value)}
          disabled={isStreaming}
        />
        <label className="block text-sm font-medium text-gray-700 mb-1">
          URL hình ảnh (tùy chọn)
        </label>
        <input
          type="url"
          className="w-full border border-gray-300 rounded-lg p-3 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="https://..."
          value={imageUrl}
          onChange={(e) => setImageUrl(e.target.value)}
          disabled={isStreaming}
        />
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Hoặc chọn tệp hình ảnh
        </label>
        <input
          type="file"
          accept="image/*"
          className="w-full border border-gray-300 rounded-lg p-3 mb-4"
          onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
          disabled={isStreaming}
        />
        <button
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold px-6 py-2 rounded-lg transition-colors"
          onClick={handleSubmit}
          disabled={isStreaming || !statement.trim()}
        >
          {isStreaming ? 'Đang kiểm tra...' : 'Kiểm tra'}
        </button>
      </div>

      {/* Error card */}
      {isError && (
        <div className="bg-red-50 border border-red-300 rounded-xl p-4 mb-6 flex items-center justify-between">
          <span className="text-red-700">Đã xảy ra lỗi. Vui lòng thử lại.</span>
          <button
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-1 rounded-lg text-sm"
            onClick={handleRetry}
          >
            Thử lại
          </button>
        </div>
      )}

      {/* Stage progress indicator (D-09) */}
      {currentStage && (
        <div className="mb-6">
          <StageIndicator
            stages={STAGES}
            currentStage={currentStage}
            completedStages={completedStages}
          />
        </div>
      )}

      {/* Debate transcript */}
      {(allTurns.length > 0 || currentTurnAgent) && (
        <div className="mb-6">
          <DebateTranscript
            turns={allTurns}
            currentTurnAgent={currentTurnAgent}
            currentTurnText={currentTurnText}
            weightBreakdown={weightBreakdown}
            showBadges={showBadges}
          />
        </div>
      )}

      {/* Verdict card */}
      {verdict && requestId && (
        <div className="mb-6">
          <VerdictCard
            verdict={verdict}
            weightBreakdown={weightBreakdown!}
            requestId={requestId}
          />
        </div>
      )}

      {/* Evidence panel — revealed with verdict (D-05) */}
      {showEvidence && (
        <div className="mb-6">
          <EvidencePanel
            evidenceReal={evidenceReal}
            evidenceFake={evidenceFake}
          />
        </div>
      )}
    </div>
  )
}
