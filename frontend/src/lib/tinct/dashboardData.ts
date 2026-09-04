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
}

export interface DashboardData {
  source: "live" | "mock"
  verified: boolean
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
}

/* eslint-disable @typescript-eslint/no-explicit-any */

function gateList(g: any): GateView[] {
  const offload = g.offload_stats
  return [
    {
      key: "canary_leakage",
      title: "Canary Leakage",
      status: g.canary_leakage?.status ?? "NOT RUN",
      detail: g.canary_leakage ? `${g.canary_leakage.leakage_rate * 100}% leaked` : "—",
    },
    {
      key: "refusal_regression",
      title: "Refusal Regression",
      status: g.refusal_regression?.status ?? "NOT RUN",
      detail: g.refusal_regression ? `${(g.refusal_regression.regression_rate * 100).toFixed(1)}% delta` : "—",
    },
    {
      key: "toxicity",
      title: "Toxicity",
      status: g.toxicity?.status ?? "NOT RUN",
      detail: g.toxicity ? `${g.toxicity.increase_factor}x factor` : "—",
    },
    {
      key: "expert_collapse",
      title: "Expert Collapse",
      status: g.expert_collapse?.status ?? "NOT RUN",
      detail: g.expert_collapse ? `Min util: ${(g.expert_collapse.min_utilization * 100).toFixed(1)}%` : "—",
    },
    {
      key: "routing_regression",
      title: "Routing Regression",
      status: g.routing_regression?.status ?? "NOT RUN",
      detail: g.routing_regression ? `${(g.routing_regression.regressed_experts ?? []).length} regressed` : "—",
    },
    {
      key: "memory_offload",
      title: "Memory Offload",
      status: offload ? "PASS" : "NOT RUN",
      detail: offload ? `${offload.cache_hits} cache hits` : "—",
    },
  ]
}

function toGb(bytes: number): number {
  return Math.round((bytes / 1024 / 1024 / 1024) * 10) / 10
}

/** Map a verified (or at least parsed) evidence bundle to the view model. */
export function bundleToDashboardData(
  runId: string,
  raw: Record<string, any>,
  verified: boolean,
): DashboardData {
  const gates = raw.safety_gates ?? {}
  const routing = gates.routing_regression
  const offload = gates.offload_stats

  const expertRouting = Array.isArray(routing?.base_utilization)
    ? routing.base_utilization.map((base: number, i: number) => ({
        expert: `Expert ${i}`,
        base: Math.round(base * 1000) / 10,
        adapter: Math.round((routing.adapter_utilization?.[i] ?? 0) * 1000) / 10,
      }))
    : []

  return {
    source: "live",
    verified,
    runId,
    baseModel: raw.model ?? "unknown",
    adapter: raw.artifacts?.adapter?.path ?? "(external adapter)",
    verdict: raw.decision === "SHIP" ? "SHIP" : "DON'T SHIP",
    timestamp: raw.created_at ?? "",
    trainingTool: raw.training_tool ?? "tinct",
    gates: gateList(gates),
    expertRouting,
    offloadStats: {
      cacheHits: offload?.cache_hits ?? 0,
      h2dStreams: offload?.h2d_streams ?? 0,
      d2hEvictions: offload?.d2h_evictions ?? 0,
      bytesH2dGb: offload ? toGb(offload.bytes_h2d) : 0,
      bytesD2hGb: offload ? toGb(offload.bytes_d2h) : 0,
      vramSavedGb: null,
    },
  }
}

/** The mock demo run, passed through the exact same view model. */
export function mockToDashboardData(): DashboardData {
  const { gates, offloadStats, ...rest } = mockRunData
  return {
    ...rest,
    source: "mock",
    verified: false,
    verdict: mockRunData.verdict === "SHIP" ? "SHIP" : "DON'T SHIP",
    trainingTool: "unsloth",
    gates: gateList(gates),
    offloadStats: {
      cacheHits: gates.memory_offload.cache_hits,
      h2dStreams: gates.memory_offload.h2d_streams,
      d2hEvictions: gates.memory_offload.d2h_evictions,
      bytesH2dGb: toGb(offloadStats.bytes_h2d),
      bytesD2hGb: toGb(offloadStats.bytes_d2h),
      vramSavedGb: offloadStats.vram_saved_gb,
    },
  }
}
