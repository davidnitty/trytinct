// Report view model for the exported PDF — pure mapping so the document's
// content is testable without a PDF renderer.
import type { DashboardData } from "@/lib/tinct/dashboardData"

export interface ReportGateRow {
  key: string
  title: string
  status: string
  metric: string
  description: string
  reason?: string
}

export interface ReportTelemetryRow {
  label: string
  value: string
}

export interface ReportRoutingRow {
  expert: string
  base: string
  adapter: string
  starved: boolean
}

export interface ReportModel {
  runId: string
  baseModel: string
  adapter: string
  completed: string
  verdict: "SHIP" | "REJECT"
  isPass: boolean
  trustLine: string
  /** Honesty markers that must travel with the exported artifact. */
  warnings: string[]
  summary: string
  gates: ReportGateRow[]
  rootCause: string
  failedGateCount: number
  telemetry: ReportTelemetryRow[]
  routing: ReportRoutingRow[]
}

export const GATE_DESCRIPTIONS: Record<string, string> = {
  canary_leakage: "Detects if the model memorized and regurgitates private data.",
  refusal_regression: "Ensures training didn't strip the model's ability to refuse harmful prompts.",
  toxicity: "Compares fine-tuned toxicity against the base model.",
  expert_collapse: "Verifies the MoE router isn't lazily sending all tokens to 1-2 experts.",
  routing_regression: "Ensures the adapter didn't break the base model's routing distribution.",
  memory_offload: "Tracks hardware utilization and VRAM savings during evaluation.",
}

const SUMMARY_PASS =
  "This model has successfully passed the tinct 6-Gate Certification Suite. It has been evaluated for data leakage, safety regression, toxicity, and structural integrity. The cryptographic evidence bundle verifies that these results have not been tampered with."

const SUMMARY_FAIL =
  "This model failed one or more safety gates during certification. It is not recommended for production deployment until the root causes (e.g., toxic training data, router collapse) are resolved."

function trustLine(data: DashboardData): string {
  if (data.source === "mock") return "Demo data — no signature"
  if (data.trusted) {
    const short = data.keyFingerprint ? ` (key ${data.keyFingerprint.slice(0, 12)}…)` : ""
    return `Ed25519 verified · pinned issuer${short}`
  }
  return "Signature mathematically valid · issuer key NOT pinned"
}

/** Warnings that must be printed on the artifact itself — a PDF escapes the
 *  dashboard, so it carries its own provenance caveats. */
function warnings(data: DashboardData): string[] {
  const lines: string[] = []
  if (data.source === "mock") {
    lines.push(
      "MOCK DATA — generated from demo data, not a real certification run. Do not circulate as evidence.",
    )
  } else if (!data.trusted) {
    lines.push(
      "UNTRUSTED ISSUER — the signing key is not in the trust store. Treat these results as unverified.",
    )
  }
  if (data.verdict === "DON'T SHIP") {
    lines.push("VERDICT IS DON'T SHIP — this model must not be deployed.")
  }
  return lines
}

function gateRows(data: DashboardData): ReportGateRow[] {
  // FAILed gates float to the top (stable within each group).
  const fails = data.gates.filter((gate) => gate.status === "FAIL")
  const rest = data.gates.filter((gate) => gate.status !== "FAIL")
  return [...fails, ...rest].map((gate) => ({
    key: gate.key,
    title: gate.title,
    status: gate.status,
    metric: gate.detail,
    description: GATE_DESCRIPTIONS[gate.key] ?? "",
    reason: gate.reason,
  }))
}

function telemetryRows(data: DashboardData): ReportTelemetryRow[] {
  const stats = data.offloadStats
  const rows: ReportTelemetryRow[] = [
    { label: "Cache hits", value: stats.cacheHits.toLocaleString() },
    { label: "H2D streams", value: String(stats.h2dStreams) },
    { label: "D2H evictions", value: String(stats.d2hEvictions) },
    { label: "H2D transferred", value: `${stats.bytesH2dGb} GB` },
    { label: "D2H transferred", value: `${stats.bytesD2hGb} GB` },
  ]
  if (stats.vramSavedGb !== null) {
    rows.unshift({ label: "VRAM saved", value: `${stats.vramSavedGb} GB` })
  }
  return rows
}

export function buildReportModel(data: DashboardData): ReportModel {
  const isPass = data.verdict === "SHIP"
  return {
    runId: data.runId,
    baseModel: data.baseModel,
    adapter: data.adapter,
    completed: data.timestamp ? new Date(data.timestamp).toLocaleString() : "—",
    verdict: isPass ? "SHIP" : "REJECT",
    isPass,
    trustLine: trustLine(data),
    warnings: warnings(data),
    summary: isPass ? SUMMARY_PASS : SUMMARY_FAIL,
    gates: gateRows(data),
    rootCause: data.rootCause,
    failedGateCount: data.failedGateCount,
    telemetry: telemetryRows(data),
    routing: data.expertRouting.map((point, index) => ({
      expert: point.expert,
      base: `${point.base}%`,
      adapter: `${point.adapter}%`,
      starved: data.starvedExperts.includes(index),
    })),
  }
}
