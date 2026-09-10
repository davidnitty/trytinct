// Dashboard view model — one shape for both real evidence bundles and the
// mock demo run, so the dashboard component never knows the difference.
// Mapping happens server-side only; the client just renders.
import { mockRunData } from "@/lib/mockData"

export type GateStatus = "PASS" | "FAIL" | "NOT RUN"

export interface GateView {
  key: string
  title: string
  status: GateStatus
  detail: string
  /** Human failure reason — set only when status is FAIL. */
  reason?: string
}

export interface DashboardData {
  source: "live" | "mock"
  verified: boolean
  trusted: boolean
  keyFingerprint: string | null
  runId: string
  baseModel: string
  adapter: string
  verdict: "SHIP" | "DON'T SHIP"
  timestamp: string
  trainingTool: string
  gates: GateView[]
  expertRouting: { expert: string; base: number; adapter: number }[]
  offloadStats: {
    cacheHits: number
    h2dStreams: number
    d2hEvictions: number
    bytesH2dGb: number
    bytesD2hGb: number
    /** Only the mock demo carries this — real bundles stream, they don't cache. */
    vramSavedGb: number | null
  }
  failedGateCount: number
  /** One-line forensic advice for the root-cause banner. Empty when all PASS. */
  rootCause: string
  /** Indices of starved experts — rendered red in the routing chart. */
  starvedExperts: number[]
  /** The expert-collapse minimum utilization threshold, in percent. */
  utilizationThresholdPct: number | null
}

/* eslint-disable @typescript-eslint/no-explicit-any */

const GATE_TITLES: Record<string, string> = {
  canary_leakage: "Canary Leakage",
  refusal_regression: "Refusal Regression",
  toxicity: "Toxicity",
  expert_collapse: "Expert Collapse",
  routing_regression: "Routing Regression",
  memory_offload: "Memory Offload",
}

const ADVICE: Record<string, string> = {
  canary_leakage: "review the training set for memorized secrets",
  refusal_regression: "rebalance the training data so safety refusals are represented",
  toxicity: "review your training data for toxic examples",
  expert_collapse: "check your MoE router configuration",
  routing_regression: "check your MoE router configuration",
}

function failureReason(key: string, g: any): string | undefined {
  if (g?.status !== "FAIL") return undefined
  switch (key) {
    case "canary_leakage":
      return `${g.canaries_leaked} of ${g.canaries_tested} canaries leaked (${(g.leakage_rate * 100).toFixed(1)}%).`
    case "refusal_regression":
      return `Adapter refusal rate shifted ${Math.abs(g.regression_rate * 100).toFixed(1)}% — beyond the ${(g.threshold * 100).toFixed(0)}% threshold.`
    case "toxicity":
      return `Adapter toxicity spiked ${g.increase_factor}x above the base model (limit ${g.threshold}x).`
    case "expert_collapse":
      return `Expert ${g.laziest_expert_id} received ${(g.min_utilization * 100).toFixed(1)}% of routing traffic (minimum ${(g.threshold * 100).toFixed(1)}%).`
    case "routing_regression":
      return `Expert(s) ${(g.regressed_experts ?? []).join(", ")} starved vs base — a relative drop of more than ${(g.relative_drop_threshold * 100).toFixed(0)}%.`
    default:
      return "Gate failed."
  }
}

function buildGates(g: any): GateView[] {
  const offload = g.offload_stats
  const gates: GateView[] = [
    {
      key: "canary_leakage",
      title: GATE_TITLES.canary_leakage,
      status: g.canary_leakage?.status ?? "NOT RUN",
      detail: g.canary_leakage ? `${g.canary_leakage.leakage_rate * 100}% leaked` : "—",
      reason: failureReason("canary_leakage", g.canary_leakage),
    },
    {
      key: "refusal_regression",
      title: GATE_TITLES.refusal_regression,
      status: g.refusal_regression?.status ?? "NOT RUN",
      detail: g.refusal_regression ? `${(g.refusal_regression.regression_rate * 100).toFixed(1)}% delta` : "—",
      reason: failureReason("refusal_regression", g.refusal_regression),
    },
    {
      key: "toxicity",
      title: GATE_TITLES.toxicity,
      status: g.toxicity?.status ?? "NOT RUN",
      detail: g.toxicity ? `${g.toxicity.increase_factor}x factor` : "—",
      reason: failureReason("toxicity", g.toxicity),
    },
    {
      key: "expert_collapse",
      title: GATE_TITLES.expert_collapse,
      status: g.expert_collapse?.status ?? "NOT RUN",
      detail: g.expert_collapse ? `Min util: ${(g.expert_collapse.min_utilization * 100).toFixed(1)}%` : "—",
      reason: failureReason("expert_collapse", g.expert_collapse),
    },
    {
      key: "routing_regression",
      title: GATE_TITLES.routing_regression,
      status: g.routing_regression?.status ?? "NOT RUN",
      detail: g.routing_regression ? `${(g.routing_regression.regressed_experts ?? []).length} regressed` : "—",
      reason: failureReason("routing_regression", g.routing_regression),
    },
    {
      key: "memory_offload",
      title: GATE_TITLES.memory_offload,
      status: offload ? "PASS" : "NOT RUN",
      detail: offload ? `${offload.cache_hits} cache hits` : "—",
    },
  ]
  // FAILED gates float to the top — forensic view first.
  return gates.sort((a, b) => {
    if (a.status === "FAIL" && b.status !== "FAIL") return -1
    if (a.status !== "FAIL" && b.status === "FAIL") return 1
    return 0
  })
}

function toGb(bytes: number): number {
  return Math.round((bytes / 1024 / 1024 / 1024) * 10) / 10
}

function summarizeFailures(gates: GateView[]): { failedGateCount: number; rootCause: string; starvedExperts: number[] } {
  const failed = gates.filter((gate) => gate.status === "FAIL")
  const advice = [...new Set(failed.map((gate) => ADVICE[gate.key]).filter(Boolean))]
  return {
    failedGateCount: failed.length,
    rootCause: failed.length
      ? `Certification failed on ${failed.length} of ${gates.length} gates. ${advice.join(", and ")}.`
      : "",
    starvedExperts: [],
  }
}

/** Map a verified (or at least parsed) evidence bundle to the view model. */
export function bundleToDashboardData(
  runId: string,
  raw: Record<string, any>,
  verification: { signatureValid: boolean; trusted: boolean; keyFingerprint: string | null },
): DashboardData {
  const gates = raw.safety_gates ?? {}
  const routing = gates.routing_regression
  const collapse = gates.expert_collapse
  const offload = gates.offload_stats

  const expertRouting = Array.isArray(routing?.base_utilization)
    ? routing.base_utilization.map((base: number, i: number) => ({
        expert: `Expert ${i}`,
        base: Math.round(base * 1000) / 10,
        adapter: Math.round((routing.adapter_utilization?.[i] ?? 0) * 1000) / 10,
      }))
    : []

  const gateViews = buildGates(gates)
  const { failedGateCount, rootCause } = summarizeFailures(gateViews)

  // Experts to paint red: a collapsed router names its laziest expert, the
  // regression gate names everything it starved vs base.
  const starvedExperts = [
    ...new Set([
      ...(collapse?.status === "FAIL" && typeof collapse.laziest_expert_id === "number"
        ? [collapse.laziest_expert_id]
        : []),
      ...((routing?.status === "FAIL" ? routing.regressed_experts : []) ?? []),
    ]),
  ]

  return {
    source: "live",
    verified: verification.signatureValid && verification.trusted,
    trusted: verification.trusted,
    keyFingerprint: verification.keyFingerprint,
    runId,
    baseModel: raw.model ?? "unknown",
    adapter: raw.artifacts?.adapter?.path ?? "(external adapter)",
    verdict: raw.decision === "SHIP" ? "SHIP" : "DON'T SHIP",
    timestamp: raw.created_at ?? "",
    trainingTool: raw.training_tool ?? "tinct",
    gates: gateViews,
    expertRouting,
    offloadStats: {
      cacheHits: offload?.cache_hits ?? 0,
      h2dStreams: offload?.h2d_streams ?? 0,
      d2hEvictions: offload?.d2h_evictions ?? 0,
      bytesH2dGb: offload ? toGb(offload.bytes_h2d) : 0,
      bytesD2hGb: offload ? toGb(offload.bytes_d2h) : 0,
      vramSavedGb: null,
    },
    failedGateCount,
    rootCause,
    starvedExperts,
    utilizationThresholdPct: collapse?.threshold != null ? collapse.threshold * 100 : null,
  }
}

/** The mock demo run, passed through the exact same view model. */
export function mockToDashboardData(): DashboardData {
  const { gates, offloadStats, ...rest } = mockRunData
  const gateViews = buildGates(gates)
  const { failedGateCount, rootCause } = summarizeFailures(gateViews)
  return {
    ...rest,
    source: "mock",
    verified: false,
    trusted: false,
    keyFingerprint: null,
    verdict: mockRunData.verdict === "SHIP" ? "SHIP" : "DON'T SHIP",
    trainingTool: "unsloth",
    gates: gateViews,
    offloadStats: {
      cacheHits: gates.memory_offload.cache_hits,
      h2dStreams: gates.memory_offload.h2d_streams,
      d2hEvictions: gates.memory_offload.d2h_evictions,
      bytesH2dGb: toGb(offloadStats.bytes_h2d),
      bytesD2hGb: toGb(offloadStats.bytes_d2h),
      vramSavedGb: offloadStats.vram_saved_gb,
    },
    failedGateCount,
    rootCause,
    starvedExperts: [],
    utilizationThresholdPct: gates.expert_collapse?.threshold != null ? gates.expert_collapse.threshold * 100 : null,
  }
}
