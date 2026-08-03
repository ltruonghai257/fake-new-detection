import { DebateTurn, WeightBreakdown } from '../App'

interface Props {
  turns: DebateTurn[]
  currentTurnAgent: string | null
  currentTurnText: string
  weightBreakdown: WeightBreakdown | null
  showBadges: boolean
}

function TurnBubble({
  turn,
  weightBreakdown,
  showBadges,
}: {
  turn: DebateTurn
  weightBreakdown: WeightBreakdown | null
  showBadges: boolean
}) {
  const isReal = turn.agent === 'real_advocate'
  const scores = showBadges
    ? weightBreakdown?.argument_scores.find(
        (s) => s.agent === turn.agent && s.round === turn.round,
      )
    : undefined

  return (
    <div className={`flex ${isReal ? 'justify-start' : 'justify-end'} mb-3`}>
      <div
        className={[
          'max-w-[75%] rounded-xl p-4',
          isReal
            ? 'bg-blue-50 border-l-4 border-blue-500'
            : 'bg-orange-50 border-r-4 border-orange-500',
        ].join(' ')}
      >
        <div className="text-xs font-semibold text-gray-500 mb-1">
          {isReal ? 'Bảo vệ' : 'Phản biện'} · Vòng {turn.round + 1}
        </div>
        <p className="text-gray-800 text-sm whitespace-pre-wrap">{turn.text}</p>
        {/* Score badges (D-07, D-08): 3 dimensions, shown only when showBadges */}
        {scores && (
          <div className="flex gap-2 mt-2 flex-wrap">
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
              Bằng chứng: {scores.factuality}
            </span>
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
              Lập luận: {scores.rebuttal_engagement}
            </span>
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
              Phản bác: {scores.evidence_grounding}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

export default function DebateTranscript({ turns, currentTurnAgent, currentTurnText, weightBreakdown, showBadges }: Props) {
  const isCurrentReal = currentTurnAgent === 'real_advocate'
  return (
    <div className="bg-white rounded-xl shadow p-4">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Tranh luận</h2>
      {turns.map((turn, idx) => (
        <TurnBubble key={idx} turn={turn} weightBreakdown={weightBreakdown} showBadges={showBadges} />
      ))}
      {/* Current turn being typed */}
      {currentTurnAgent && (
        <div className={`flex ${isCurrentReal ? 'justify-start' : 'justify-end'} mb-3`}>
          <div
            className={[
              'max-w-[75%] rounded-xl p-4',
              isCurrentReal
                ? 'bg-blue-50 border-l-4 border-blue-500'
                : 'bg-orange-50 border-r-4 border-orange-500',
            ].join(' ')}
          >
            <div className="text-xs font-semibold text-gray-500 mb-1">
              {isCurrentReal ? 'Bảo vệ' : 'Phản biện'} · Đang viết...
            </div>
            <p className="text-gray-800 text-sm whitespace-pre-wrap">
              {currentTurnText}
              <span className="animate-pulse">▌</span>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
