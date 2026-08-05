import { useState, useEffect, useRef } from 'react';
import StageIndicator from './components/StageIndicator';
import DebateTranscript from './components/DebateTranscript';
import VerdictCard from './components/VerdictCard';
import EvidencePanel from './components/EvidencePanel';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Verdict {
    label: string;
    verdict_binary: 'REAL' | 'FAKE';
    verdict_label_vi: 'Thật' | 'Giả';
    confidence: number;
    rationale: string;
    citations: string[];
    recommendation: string;
}

export interface ArgumentScore {
    agent: string;
    round: number;
    factuality: number;
    rebuttal_engagement: number;
    evidence_grounding: number;
}

export interface EvidenceBreakdown {
    tier_score: number;
    count_score: number;
    consistency_score: number;
    trusted_count: number;
    total_real: number;
    total_fake: number;
    total_evidence: number;
}

export interface WeightBreakdown {
    phobert: number;
    coolant: number;
    evidence: number;
    argument_scores: ArgumentScore[];
    phobert_label?: string | null;
    phobert_probabilities?: Record<string, number> | null;
    coolant_label?: string | null;
    coolant_probabilities?: Record<string, number> | null;
    evidence_breakdown?: EvidenceBreakdown | null;
}

export interface Evidence {
    title: string;
    url: string;
    snippet: string;
    source_tier: 'trusted' | 'flagged' | 'social' | 'unknown';
}

export interface DebateTurn {
    agent: 'real_advocate' | 'fake_advocate';
    round: number;
    text: string;
    timestamp: string;
    error?: string;
}

// ── Stage constants (D-10) ────────────────────────────────────────────────────

export const STAGES = [
    'evidence_retrieval',
    'reranking',
    'verification',
    'debate',
    'verdict',
] as const;
export type StageName = (typeof STAGES)[number];

export const STAGE_LABELS: Record<StageName, string> = {
    evidence_retrieval: 'Tìm bằng chứng',
    reranking: 'Xếp hạng bằng chứng',
    verification: 'Kiểm định mô hình',
    debate: 'Tranh luận',
    verdict: 'Phán quyết',
};

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
    // Form state
    const [statement, setStatement] = useState('');
    const [imageUrl, setImageUrl] = useState('');
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);

    // Load image preview when URL changes (real-time, debounced)
    useEffect(() => {
        const timer = setTimeout(() => {
            if (imageUrl.trim()) {
                setImagePreview(imageUrl.trim());
            } else {
                setImagePreview(null);
            }
        }, 300); // 300ms debounce to avoid flickering while typing
        return () => clearTimeout(timer);
    }, [imageUrl]);

    // Load preview when file selected
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] ?? null;
        setImageFile(file);
        if (file) {
            const reader = new FileReader();
            reader.onload = () => setImagePreview(reader.result as string);
            reader.readAsDataURL(file);
        } else {
            setImagePreview(null);
        }
    };

    // Analysis state
    const [requestId, setRequestId] = useState<string | null>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isError, setIsError] = useState(false);

    // Pipeline stage state (D-09)
    const [currentStage, setCurrentStage] = useState<StageName | null>(null);
    const [completedStages, setCompletedStages] = useState<StageName[]>([]);

    // Debate transcript state
    const [allTurns, setAllTurns] = useState<DebateTurn[]>([]);
    const [currentTurnAgent, setCurrentTurnAgent] = useState<string | null>(
        null
    );
    const [currentTurnRound, setCurrentTurnRound] = useState<number>(0);
    const currentTurnTextRef = useRef('');
    const [currentTurnText, setCurrentTurnText] = useState('');

    // Verdict + evidence state (revealed together at verdict event, D-05/D-07)
    const [verdict, setVerdict] = useState<Verdict | null>(null);
    const [weightBreakdown, setWeightBreakdown] =
        useState<WeightBreakdown | null>(null);
    const [evidenceReal, setEvidenceReal] = useState<Evidence[]>([]);
    const [evidenceFake, setEvidenceFake] = useState<Evidence[]>([]);
    const [showEvidence, setShowEvidence] = useState(false);
    const [showBadges, setShowBadges] = useState(false);
    const [stageLogs, setStageLogs] = useState<
        { stage: string; message: string }[]
    >([]);

    // SSE: StrictMode-safe EventSource lifecycle (DEMO-04)
    useEffect(() => {
        if (!requestId) return;

        const es = new EventSource(
            `http://localhost:8000/api/stream/${requestId}`
        );
        setIsStreaming(true);

        es.addEventListener('stage_start', e => {
            const data = JSON.parse(e.data) as { type: string; name: string };
            const name = data.name as StageName;
            setCurrentStage(name);
            setCompletedStages(prev =>
                prev.includes(name) ? prev : [...prev, name]
            );
        });

        es.addEventListener('stage_log', e => {
            const data = JSON.parse(e.data) as {
                type: string;
                stage: string;
                message: string;
            };
            setStageLogs(prev => [
                ...prev,
                { stage: data.stage, message: data.message },
            ]);
        });

        es.addEventListener('model_results', e => {
            const data = JSON.parse(e.data) as {
                type: string;
                results: Array<{
                    model: string;
                    label: string;
                    confidence: number;
                    probabilities: Record<string, number>;
                }>;
            };
            setModelResults(data.results);
        });

        es.addEventListener('turn_start', e => {
            const data = JSON.parse(e.data) as {
                type: string;
                agent: string;
                round: number;
            };
            setCurrentTurnAgent(data.agent);
            setCurrentTurnRound(data.round);
            currentTurnTextRef.current = '';
            setCurrentTurnText('');
        });

        es.addEventListener('chunk', e => {
            const data = JSON.parse(e.data) as { type: string; text: string };
            currentTurnTextRef.current += data.text;
            setCurrentTurnText(currentTurnTextRef.current);
        });

        es.addEventListener('turn_end', e => {
            const data = JSON.parse(e.data) as {
                type: string;
                agent: string;
                round: number;
            };
            const finalText = currentTurnTextRef.current;
            setAllTurns(prev => [
                ...prev,
                {
                    agent: data.agent as 'real_advocate' | 'fake_advocate',
                    round: data.round,
                    text: finalText,
                    timestamp: new Date().toISOString(),
                },
            ]);
            setCurrentTurnAgent(null);
            setCurrentTurnRound(0);
            currentTurnTextRef.current = '';
            setCurrentTurnText('');
        });

        es.addEventListener('verdict', e => {
            const data = JSON.parse(e.data) as {
                type: string;
                verdict: Verdict;
                weight_breakdown: WeightBreakdown;
                evidence_real: Evidence[];
                evidence_fake: Evidence[];
                debate_turns: DebateTurn[];
            };
            setVerdict(data.verdict);
            setWeightBreakdown(data.weight_breakdown);
            setEvidenceReal(data.evidence_real ?? []);
            setEvidenceFake(data.evidence_fake ?? []);
            setShowEvidence(true); // D-05
            setShowBadges(true); // D-07
            setIsStreaming(false);
            es.close();
        });

        es.onerror = () => {
            setIsError(true);
            setIsStreaming(false);
            es.close();
        };

        return () => es.close(); // StrictMode cleanup (DEMO-04)
    }, [requestId]);

    const handleSubmit = async () => {
        if (!statement.trim()) return;
        // Reset state for new analysis
        setIsError(false);
        setCurrentStage(null);
        setCompletedStages([]);
        setAllTurns([]);
        setCurrentTurnAgent(null);
        setCurrentTurnRound(0);
        currentTurnTextRef.current = '';
        setCurrentTurnText('');
        setVerdict(null);
        setWeightBreakdown(null);
        setEvidenceReal([]);
        setEvidenceFake([]);
        setShowEvidence(false);
        setShowBadges(false);
        setStageLogs([]);

        const fd = new FormData();
        fd.append('statement', statement);
        if (imageUrl.trim()) fd.append('image_url', imageUrl.trim());
        if (imageFile) fd.append('image_file', imageFile);

        try {
            const res = await fetch('http://localhost:8000/api/analyze', {
                method: 'POST',
                body: fd,
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const json = (await res.json()) as { request_id: string };
            setRequestId(json.request_id);
        } catch {
            setIsError(true);
        }
    };

    const handleRetry = () => {
        setIsError(false);
        setRequestId(null);
    };

    return (
        <div className="min-h-screen bg-gray-50 p-6 max-w-4xl mx-auto">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">
                Kiểm tra tin giả
            </h1>

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
                    onChange={e => setStatement(e.target.value)}
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
                    onChange={e => setImageUrl(e.target.value)}
                    disabled={isStreaming}
                />
                <label className="block text-sm font-medium text-gray-700 mb-1">
                    Hoặc chọn tệp hình ảnh
                </label>
                <input
                    type="file"
                    accept="image/*"
                    className="w-full border border-gray-300 rounded-lg p-3 mb-4"
                    onChange={handleFileChange}
                    disabled={isStreaming}
                />
                {/* Image preview */}
                {imagePreview && (
                    <div className="mb-4">
                        <div className="text-sm font-medium text-gray-700 mb-1">
                            Xem trước hình ảnh
                        </div>
                        <img
                            src={imagePreview}
                            alt="Preview"
                            className="max-h-48 rounded-lg border border-gray-200 object-contain"
                        />
                    </div>
                )}
                <button
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold px-6 py-2 rounded-lg transition-colors"
                    onClick={handleSubmit}
                    disabled={isStreaming || !statement.trim()}>
                    {isStreaming ? 'Đang kiểm tra...' : 'Kiểm tra'}
                </button>
            </div>

            {/* Error card */}
            {isError && (
                <div className="bg-red-50 border border-red-300 rounded-xl p-4 mb-6 flex items-center justify-between">
                    <span className="text-red-700">
                        Đã xảy ra lỗi. Vui lòng thử lại.
                    </span>
                    <button
                        className="bg-red-600 hover:bg-red-700 text-white px-4 py-1 rounded-lg text-sm"
                        onClick={handleRetry}>
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

            {/* Stage logs — running summary of what each step collected */}
            {stageLogs.length > 0 && (
                <div className="bg-white rounded-xl shadow p-4 mb-6">
                    <div className="text-sm font-medium text-gray-700 mb-2">
                        Nhật ký xử lý
                    </div>
                    <ul className="space-y-1 text-sm text-gray-600 font-mono">
                        {stageLogs.map((log, i) => (
                            <li key={i} className="flex gap-2">
                                <span className="text-gray-400 shrink-0">
                                    [{log.stage}]
                                </span>
                                <span>{log.message}</span>
                            </li>
                        ))}
                    </ul>
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
    );
}
