import { useState } from 'react'
import { Evidence } from '../App'

interface Props {
  evidenceReal: Evidence[]
  evidenceFake: Evidence[]
}

const TIER_CLASSES: Record<Evidence['source_tier'], string> = {
  trusted: 'bg-green-100 text-green-700',
  flagged: 'bg-orange-100 text-orange-700',
  social:  'bg-blue-100 text-blue-700',
  unknown: 'bg-gray-100 text-gray-600',
}

const TIER_LABELS: Record<Evidence['source_tier'], string> = {
  trusted: 'Đáng tin',
  flagged: 'Nghi vấn',
  social:  'Mạng xã hội',
  unknown: 'Không rõ',
}

function EvidenceItem({ item }: { item: Evidence }) {
  return (
    <div className="border border-gray-200 rounded-lg p-3">
      <div className="flex items-start justify-between gap-2 mb-1">
        <a
          href={item.url}
          target="_blank"
          rel="noreferrer"
          className="text-blue-600 hover:underline text-sm font-medium line-clamp-2"
        >
          {item.title || item.url}
        </a>
        <span className={`text-xs px-2 py-0.5 rounded-full whitespace-nowrap ${TIER_CLASSES[item.source_tier]}`}>
          {TIER_LABELS[item.source_tier]}
        </span>
      </div>
      {item.snippet && (
        <p className="text-gray-600 text-xs line-clamp-3">{item.snippet}</p>
      )}
    </div>
  )
}

export default function EvidencePanel({ evidenceReal, evidenceFake }: Props) {
  const [activeTab, setActiveTab] = useState<'real' | 'fake'>('real')
  const items = activeTab === 'real' ? evidenceReal : evidenceFake

  return (
    <div className="bg-white rounded-xl shadow p-4">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Bằng chứng</h2>
      {/* Tabs (D-06) */}
      <div className="flex gap-2 mb-4">
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'real' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
          onClick={() => setActiveTab('real')}
        >
          Nguồn ủng hộ ({evidenceReal.length})
        </button>
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'fake' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
          onClick={() => setActiveTab('fake')}
        >
          Nguồn phản bác ({evidenceFake.length})
        </button>
      </div>
      {items.length === 0 ? (
        <p className="text-gray-400 text-sm text-center py-4">Không có bằng chứng</p>
      ) : (
        <div className="space-y-2">
          {items.map((item, idx) => (
            <EvidenceItem key={idx} item={item} />
          ))}
        </div>
      )}
    </div>
  )
}
