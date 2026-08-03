import { STAGE_LABELS, STAGES, StageName } from '../App'

interface Props {
  stages: readonly StageName[]
  currentStage: StageName | null
  completedStages: StageName[]
}

export default function StageIndicator({ stages, currentStage, completedStages }: Props) {
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {stages.map((stage, idx) => {
        const isActive = currentStage === stage
        const isDone = completedStages.includes(stage) && !isActive
        return (
          <div key={stage} className="flex items-center gap-1">
            <span
              className={[
                'px-3 py-1 rounded-full text-sm font-medium transition-colors',
                isActive
                  ? 'bg-blue-600 text-white ring-2 ring-blue-300'
                  : isDone
                  ? 'bg-green-100 text-green-700'
                  : 'bg-gray-200 text-gray-500',
              ].join(' ')}
            >
              {isDone && '✓ '}
              {STAGE_LABELS[stage]}
            </span>
            {idx < stages.length - 1 && (
              <span className="text-gray-400 text-xs">→</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
