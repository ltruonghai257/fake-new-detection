import { Verdict, WeightBreakdown } from '../App'

interface Props {
  verdict: Verdict
  weightBreakdown: WeightBreakdown
  requestId: string
}

export default function VerdictCard({ verdict, weightBreakdown, requestId }: Props) {
  const isReal = verdict.verdict_binary === 'REAL'
  const confidencePct = Math.round(verdict.confidence * 100)

  return (
    <div className="bg-white rounded-xl shadow p-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Phán quyết</h2>

      {/* Label + confidence */}
      <div className="flex items-center gap-4 mb-4">
        <span className={`text-4xl font-bold ${isReal ? 'text-green-600' : 'text-red-600'}`}>
          {verdict.verdict_label_vi}
        </span>
        <span className="text-gray-500 text-sm">{verdict.label}</span>
        <span className="ml-auto text-gray-600 font-medium">{confidencePct}% tin cậy</span>
      </div>

      {/* Confidence gauge */}
      <div className="w-full bg-gray-200 rounded-full h-3 mb-6">
        <div
          className={`h-3 rounded-full transition-all ${isReal ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: `${confidencePct}%` }}
        />
      </div>

      {/* 30/30/40 weight breakdown bar (DEMO-03) */}
      <div className="mb-6">
        <div className="text-sm font-medium text-gray-700 mb-2">Phân bổ trọng số</div>
        <div className="flex rounded-lg overflow-hidden h-8">
          <div
            className="bg-purple-400 flex items-center justify-center text-white text-xs font-medium"
            style={{ flex: weightBreakdown.phobert }}
          >
            PhoBERT {Math.round(weightBreakdown.phobert * 100)}%
          </div>
          <div
            className="bg-teal-400 flex items-center justify-center text-white text-xs font-medium"
            style={{ flex: weightBreakdown.coolant }}
          >
            COOLANT {Math.round(weightBreakdown.coolant * 100)}%
          </div>
          <div
            className="bg-amber-400 flex items-center justify-center text-white text-xs font-medium"
            style={{ flex: weightBreakdown.evidence }}
          >
            Bằng chứng {Math.round(weightBreakdown.evidence * 100)}%
          </div>
        </div>
      </div>

      {/* Rationale */}
      <p className="text-gray-700 mb-3">{verdict.rationale}</p>
      <p className="text-gray-500 text-sm mb-4">{verdict.recommendation}</p>

      {/* Citations */}
      {verdict.citations.length > 0 && (
        <div className="mb-4">
          <div className="text-sm font-medium text-gray-700 mb-1">Nguồn tham khảo</div>
          <ul className="list-disc list-inside space-y-1">
            {verdict.citations.map((url, i) => (
              <li key={i}>
                <a href={url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline text-sm truncate">
                  {url}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Download buttons (DEMO-03) */}
      <div className="flex gap-3 flex-wrap">
        <a
          href={`http://localhost:8000/api/download/debate/${requestId}`}
          download
          className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          Tải bản ghi tranh luận
        </a>
        <a
          href={`http://localhost:8000/api/download/verdict/${requestId}`}
          download
          className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          Tải bản phán quyết
        </a>
      </div>
    </div>
  )
}
