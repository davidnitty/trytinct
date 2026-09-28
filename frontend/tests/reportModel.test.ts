// Report view model: gate ordering, provenance warnings, telemetry shaping.
// Pure mapping — the PDF renderer itself is exercised in pdfRender.test.tsx.
import { describe, expect, it } from "vitest"

import type { DashboardData } from "@/lib/tinct/dashboardData"
import { GATE_DESCRIPTIONS, buildReportModel } from "@/lib/tinct/reportModel"

function data(overrides: Partial<DashboardData> = {}): DashboardData {
  return {
    source: "live",
    verified: true,
    trusted: true,
    keyFingerprint: "ffbd0c0567af392ad1c5b082fca45cd4dabd0bb7403d9f758e8fb5e9ab449e17",
    runId: "cert_20260904_143022",
    baseModel: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    adapter: "runs/cert_20260904_143022/adapter",
    verdict: "SHIP",
    timestamp: "2026-09-04T14:35:10+00:00",
    trainingTool: "unsloth",
    gates: [
      { key: "canary_leakage", title: "Canary Leakage", status: "PASS", detail: "0% leaked" },
      { key: "refusal_regression", title: "Refusal Regression", status: "PASS", detail: "-3.5% delta" },
      { key: "toxicity", title: "Toxicity", status: "FAIL", detail: "3.4x factor", reason: "Adapter toxicity spiked 3.4x." },
      { key: "expert_collapse", title: "Expert Collapse", status: "PASS", detail: "Min util: 11.0%" },
      { key: "routing_regression", title: "Routing Regression", status: "PASS", detail: "0 regressed" },
      { key: "memory_offload", title: "Memory Offload", status: "NOT RUN", detail: "—" },
    ],
    expertRouting: [
      { expert: "Expert 0", base: 12.5, adapter: 13.1 },
      { expert: "Expert 6", base: 12.6, adapter: 4.1 },
    ],
    offloadStats: {
      cacheHits: 8450,
      h2dStreams: 142,
      d2hEvictions: 134,
      bytesH2dGb: 4,
      bytesD2hGb: 3.9,
      vramSavedGb: null,
    },
    failedGateCount: 1,
    rootCause: "review your training data for toxic examples.",
    starvedExperts: [1],
    utilizationThresholdPct: 1,
    telemetry: [],
    ...overrides,
  }
}

describe("gate rows", () => {
  it("floats FAILed gates to the top, preserving order within groups", () => {
    const rows = buildReportModel(data()).gates
    expect(rows[0].key).toBe("toxicity")
    expect(rows[0].status).toBe("FAIL")
    expect(rows.slice(1).map((r) => r.key)).toEqual([
      "canary_leakage",
      "refusal_regression",
      "expert_collapse",
      "routing_regression",
      "memory_offload",
    ])
  })

  it("carries the failure reason through", () => {
    const toxicity = buildReportModel(data()).gates.find((r) => r.key === "toxicity")
    expect(toxicity?.reason).toBe("Adapter toxicity spiked 3.4x.")
  })

  it("has a description for every gate the product can emit", () => {
    for (const key of [
      "canary_leakage",
      "refusal_regression",
      "toxicity",
      "expert_collapse",
      "routing_regression",
      "memory_offload",
    ]) {
      expect(GATE_DESCRIPTIONS[key]).toBeTruthy()
    }
  })
})

describe("verdict and provenance", () => {
  it("labels a failing verdict REJECT", () => {
    const report = buildReportModel(data({ verdict: "DON'T SHIP", failedGateCount: 1 }))
    expect(report.verdict).toBe("REJECT")
    expect(report.isPass).toBe(false)
    expect(report.warnings.some((w) => w.includes("DON'T SHIP"))).toBe(true)
  })

  it("warns that mock data must not circulate as evidence", () => {
    const report = buildReportModel(data({ source: "mock", verified: false, trusted: false }))
    expect(report.warnings.some((w) => w.startsWith("MOCK DATA"))).toBe(true)
    expect(report.trustLine).toContain("Demo data")
  })

  it("warns on an unpinned issuer even when the math verifies", () => {
    const report = buildReportModel(data({ trusted: false, verified: false }))
    expect(report.warnings.some((w) => w.startsWith("UNTRUSTED ISSUER"))).toBe(true)
    expect(report.trustLine).toContain("NOT pinned")
  })

  it("emits no warnings for a trusted SHIP", () => {
    expect(buildReportModel(data()).warnings).toEqual([])
    expect(buildReportModel(data()).trustLine).toContain("Ed25519 verified")
  })
})

describe("telemetry and routing", () => {
  it("omits the VRAM row when the stats do not carry it", () => {
    const labels = buildReportModel(data()).telemetry.map((r) => r.label)
    expect(labels).not.toContain("VRAM saved")
    expect(labels).toContain("Cache hits")
  })

  it("leads with VRAM saved when present", () => {
    const stats = { ...data().offloadStats, vramSavedGb: 68.4 }
    const rows = buildReportModel(data({ offloadStats: stats })).telemetry
    expect(rows[0]).toEqual({ label: "VRAM saved", value: "68.4 GB" })
  })

  it("flags starved experts by index", () => {
    const routing = buildReportModel(data()).routing
    expect(routing[0].starved).toBe(false)
    expect(routing[1].starved).toBe(true)
    expect(routing[1].adapter).toBe("4.1%")
  })

  it("handles a dense-model run with no routing data", () => {
    expect(buildReportModel(data({ expertRouting: [] })).routing).toEqual([])
  })
})
