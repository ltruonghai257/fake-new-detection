import { useState } from 'react';
import { Verdict, WeightBreakdown, EvidenceBreakdown } from '../App';

interface Props {
    verdict: Verdict;
    weightBreakdown: WeightBreakdown;
    requestId: string;
}

type DetailPanel = 'phobert' | 'coolant' | 'evidence' | null;

export default function VerdictCard({
    verdict,
    weightBreakdown,
    requestId,
}: Props) {
    const isReal = verdict.verdict_binary === 'REAL';
    const confidencePct = Math.round(verdict.confidence * 100);
    const [openPanel, setOpenPanel] = useState<DetailPanel>(null);

    const toggle = (panel: DetailPanel) =>
        setOpenPanel(prev => (prev === panel ? null : panel));

    return (
        <div className="bg-white rounded-xl shadow p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
                Phán quyết
            </h2>

            {/* Label + confidence */}
            <div className="flex items-center gap-4 mb-4">
                <span
                    className={`text-4xl font-bold ${
                        isReal ? 'text-green-600' : 'text-red-600'
                    }`}>
                    {verdict.verdict_label_vi}
                </span>
                <span className="text-gray-500 text-sm">{verdict.label}</span>
                <span className="ml-auto text-gray-600 font-medium">
                    {confidencePct}% tin cậy
                </span>
            </div>

            {/* Confidence gauge */}
            <div className="w-full bg-gray-200 rounded-full h-3 mb-6">
                <div
                    className={`h-3 rounded-full transition-all ${
                        isReal ? 'bg-green-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${confidencePct}%` }}
                />
            </div>

            {/* 30/30/40 weight breakdown bar — click segment to expand details */}
            <div className="mb-6">
                <div className="text-sm font-medium text-gray-700 mb-2">
                    Phân bổ trọng số{' '}
                    <span className="text-gray-400 font-normal text-xs">
                        (nhấn vào từng cột để xem chi tiết)
                    </span>
                </div>
                <div className="flex rounded-lg h-10">
                    <button
                        type="button"
                        onClick={() => toggle('phobert')}
                        className={`group relative bg-purple-400 flex flex-col items-center justify-center text-white text-xs font-medium leading-tight cursor-pointer first:rounded-l-lg transition-opacity ${
                            openPanel === 'phobert'
                                ? 'ring-2 ring-purple-600 ring-offset-1'
                                : ''
                        }`}
                        style={{ flex: weightBreakdown.phobert }}>
                        <span>
                            PhoBERT {Math.round(weightBreakdown.phobert * 100)}%
                        </span>
                        {weightBreakdown.phobert_label && (
                            <span className="text-purple-100 text-[10px]">
                                {weightBreakdown.phobert_label}
                            </span>
                        )}
                    </button>
                    <button
                        type="button"
                        onClick={() => toggle('coolant')}
                        className={`group relative bg-teal-400 flex flex-col items-center justify-center text-white text-xs font-medium leading-tight cursor-pointer transition-opacity ${
                            openPanel === 'coolant'
                                ? 'ring-2 ring-teal-600 ring-offset-1'
                                : ''
                        }`}
                        style={{ flex: weightBreakdown.coolant }}>
                        <span>
                            COOLANT {Math.round(weightBreakdown.coolant * 100)}%
                        </span>
                        {weightBreakdown.coolant_label && (
                            <span className="text-teal-100 text-[10px]">
                                {weightBreakdown.coolant_label}
                            </span>
                        )}
                    </button>
                    <button
                        type="button"
                        onClick={() => toggle('evidence')}
                        className={`group relative bg-amber-400 flex items-center justify-center text-white text-xs font-medium cursor-pointer last:rounded-r-lg transition-opacity ${
                            openPanel === 'evidence'
                                ? 'ring-2 ring-amber-600 ring-offset-1'
                                : ''
                        }`}
                        style={{ flex: weightBreakdown.evidence }}>
                        Bằng chứng {Math.round(weightBreakdown.evidence * 100)}%
                    </button>
                </div>
                <p className="text-[11px] text-gray-400 mt-1">
                    Điểm cuối = 0.30 × PhoBERT + 0.30 × COOLANT + 0.40 × Bằng
                    chứng
                </p>

                {/* ── Detail panels ──────────────────────────────────────────── */}
                {openPanel === 'phobert' && (
                    <PhobertDetail
                        confidence={weightBreakdown.phobert}
                        label={weightBreakdown.phobert_label}
                        probabilities={weightBreakdown.phobert_probabilities}
                    />
                )}
                {openPanel === 'coolant' && (
                    <CoolantDetail
                        confidence={weightBreakdown.coolant}
                        label={weightBreakdown.coolant_label}
                        probabilities={weightBreakdown.coolant_probabilities}
                    />
                )}
                {openPanel === 'evidence' && (
                    <EvidenceDetail
                        credScore={weightBreakdown.evidence}
                        breakdown={weightBreakdown.evidence_breakdown}
                    />
                )}
            </div>

            {/* Rationale */}
            <p className="text-gray-700 mb-3">{verdict.rationale}</p>
            <p className="text-gray-500 text-sm mb-4">
                {verdict.recommendation}
            </p>

            {/* Citations */}
            {verdict.citations.length > 0 && (
                <div className="mb-4">
                    <div className="text-sm font-medium text-gray-700 mb-1">
                        Nguồn tham khảo
                    </div>
                    <ul className="list-disc list-inside space-y-1">
                        {verdict.citations.map((url, i) => (
                            <li key={i}>
                                <a
                                    href={url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-blue-600 hover:underline text-sm truncate">
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
                    className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                    Tải bản ghi tranh luận
                </a>
                <a
                    href={`http://localhost:8000/api/download/verdict/${requestId}`}
                    download
                    className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                    Tải bản phán quyết
                </a>
            </div>
        </div>
    );
}

// ── Detail sub-components ───────────────────────────────────────────────────

function PhobertDetail({
    confidence,
    label,
    probabilities,
}: {
    confidence: number;
    label?: string | null;
    probabilities?: Record<string, number> | null;
}) {
    return (
        <div className="mt-3 bg-purple-50 border border-purple-200 rounded-lg p-4 text-sm">
            <div className="font-medium text-purple-800 mb-2">
                PhoBERT — chi tiết điểm {Math.round(confidence * 100)}%
            </div>
            <p className="text-gray-600 mb-3">
                PhoBERT nhận claim + bằng chứng đã truy xuất, xuất ra 3 logit
                cho 3 nhãn. Áp dụng{' '}
                <code className="bg-purple-100 px-1 rounded">softmax</code> để
                được xác suất, nhãn có xác suất cao nhất được chọn và xác suất
                đó chính là độ tin cậy.
            </p>
            {probabilities && (
                <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-500">
                        Phân phối xác suất:
                    </div>
                    {Object.entries(probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .map(([lbl, prob]) => (
                            <div key={lbl} className="flex items-center gap-2">
                                <span
                                    className={`w-24 text-xs font-medium ${
                                        lbl === label
                                            ? 'text-purple-700'
                                            : 'text-gray-500'
                                    }`}>
                                    {lbl}
                                    {lbl === label && ' ✓'}
                                </span>
                                <div className="flex-1 bg-purple-100 rounded-full h-4 overflow-hidden">
                                    <div
                                        className="bg-purple-500 h-4 rounded-full"
                                        style={{ width: `${prob * 100}%` }}
                                    />
                                </div>
                                <span className="text-xs text-gray-600 w-12 text-right">
                                    {(prob * 100).toFixed(1)}%
                                </span>
                            </div>
                        ))}
                </div>
            )}
            <p className="text-xs text-gray-400 mt-3">
                Công thức: confidence = max(softmax(logits)) · Đóng góp 30% vào
                điểm cuối.
            </p>
        </div>
    );
}

function CoolantDetail({
    confidence,
    label,
    probabilities,
}: {
    confidence: number;
    label?: string | null;
    probabilities?: Record<string, number> | null;
}) {
    return (
        <div className="mt-3 bg-teal-50 border border-teal-200 rounded-lg p-4 text-sm">
            <div className="font-medium text-teal-800 mb-2">
                COOLANT — chi tiết điểm {Math.round(confidence * 100)}%
            </div>
            <p className="text-gray-600 mb-3">
                COOLANT là mô hình đa phương thức: nhận claim (text) + hình ảnh,
                xuất ra 2 logit cho REAL / FAKE. Áp dụng{' '}
                <code className="bg-teal-100 px-1 rounded">softmax</code> để
                được xác suất, nhãn cao nhất được chọn.
            </p>
            {probabilities && (
                <div className="space-y-2">
                    <div className="text-xs font-medium text-gray-500">
                        Phân phối xác suất:
                    </div>
                    {Object.entries(probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .map(([lbl, prob]) => (
                            <div key={lbl} className="flex items-center gap-2">
                                <span
                                    className={`w-20 text-xs font-medium ${
                                        lbl === label
                                            ? 'text-teal-700'
                                            : 'text-gray-500'
                                    }`}>
                                    {lbl}
                                    {lbl === label && ' ✓'}
                                </span>
                                <div className="flex-1 bg-teal-100 rounded-full h-4 overflow-hidden">
                                    <div
                                        className="bg-teal-500 h-4 rounded-full"
                                        style={{ width: `${prob * 100}%` }}
                                    />
                                </div>
                                <span className="text-xs text-gray-600 w-12 text-right">
                                    {(prob * 100).toFixed(1)}%
                                </span>
                            </div>
                        ))}
                </div>
            )}
            <p className="text-xs text-gray-400 mt-3">
                Công thức: confidence = max(softmax(detection_logits)) · Đóng
                góp 30% vào điểm cuối.
            </p>
        </div>
    );
}

function EvidenceDetail({
    credScore,
    breakdown,
}: {
    credScore: number;
    breakdown?: EvidenceBreakdown | null;
}) {
    if (!breakdown) {
        return (
            <div className="mt-3 bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-gray-500">
                Không có dữ liệu chi tiết.
            </div>
        );
    }
    const tierPct = Math.round(breakdown.tier_score * 100);
    const countPct = Math.round(breakdown.count_score * 100);
    const consPct = Math.round(breakdown.consistency_score * 100);
    const credPct = Math.round(credScore * 100);

    return (
        <div className="mt-3 bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm">
            <div className="font-medium text-amber-800 mb-2">
                Bằng chứng — chi tiết điểm {credPct}%
            </div>
            <p className="text-gray-600 mb-3">
                Điểm uy tín ={' '}
                <code className="bg-amber-100 px-1 rounded">
                    0.40 × tier + 0.30 × count + 0.30 × consistency
                </code>
            </p>

            <div className="space-y-3">
                {/* Tier score */}
                <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="font-medium text-gray-600">
                            Tier score (tỷ lệ nguồn tin cậy)
                        </span>
                        <span className="text-gray-500">{tierPct}%</span>
                    </div>
                    <div className="bg-amber-100 rounded-full h-3 overflow-hidden">
                        <div
                            className="bg-amber-500 h-3 rounded-full"
                            style={{ width: `${tierPct}%` }}
                        />
                    </div>
                    <p className="text-[11px] text-gray-400 mt-1">
                        {breakdown.trusted_count} / {breakdown.total_real} nguồn
                        ủng hộ đến từ domain tin cậy (vnexpress, thanhnien,
                        tuoitre, dantri)
                    </p>
                </div>

                {/* Count score */}
                <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="font-medium text-gray-600">
                            Count score (số lượng bằng chứng)
                        </span>
                        <span className="text-gray-500">{countPct}%</span>
                    </div>
                    <div className="bg-amber-100 rounded-full h-3 overflow-hidden">
                        <div
                            className="bg-amber-500 h-3 rounded-full"
                            style={{ width: `${countPct}%` }}
                        />
                    </div>
                    <p className="text-[11px] text-gray-400 mt-1">
                        {breakdown.total_evidence} bằng chứng (
                        {breakdown.total_real} ủng hộ + {breakdown.total_fake}{' '}
                        phản bác) · bão hòa ở 5 nguồn
                    </p>
                </div>

                {/* Consistency score */}
                <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="font-medium text-gray-600">
                            Consistency score (mức độ nhất quán)
                        </span>
                        <span className="text-gray-500">{consPct}%</span>
                    </div>
                    <div className="bg-amber-100 rounded-full h-3 overflow-hidden">
                        <div
                            className="bg-amber-500 h-3 rounded-full"
                            style={{ width: `${consPct}%` }}
                        />
                    </div>
                    <p className="text-[11px] text-gray-400 mt-1">
                        Trung bình cosine similarity giữa embedding bằng chứng
                        và embedding claim (sentence-transformers), floor 0.1
                    </p>
                </div>

                {/* Final calc */}
                <div className="pt-2 border-t border-amber-200">
                    <div className="flex justify-between text-xs">
                        <span className="font-medium text-amber-700">
                            Điểm uy tín tổng hợp
                        </span>
                        <span className="font-bold text-amber-700">
                            {credPct}%
                        </span>
                    </div>
                    <p className="text-[11px] text-gray-400 mt-1">
                        = 0.40 × {tierPct}% + 0.30 × {countPct}% + 0.30 ×{' '}
                        {consPct}% = {credPct}% · Đóng góp 40% vào điểm cuối
                    </p>
                </div>
            </div>
        </div>
    );
}
