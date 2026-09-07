// ── Shared types & stage constants ───────────────────────────────────────────

export interface VerdictExplanation {
    model_summary: string;
    debate_winner: string;
    evidence_summary: string;
    confidence_breakdown: {
        phobert: number;
        coolant: number;
        evidence: number;
        debate: number;
    };
}

export interface Verdict {
    label: string;
    verdict_binary: 'REAL' | 'FAKE' | 'NEI';
    verdict_label_vi: 'Thật' | 'Giả' | 'Chưa xác thực';
    confidence: number;
    rationale: string;
    citations: string[];
    recommendation: string;
    explanation?: VerdictExplanation | null;
    model_detail?: Record<string, { label: string; confidence: number; probabilities: Record<string, number> }> | null;
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
    verdict?: 'REAL' | 'FAKE' | null;
    confidence?: number | null;
    concession?: string | null;
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
