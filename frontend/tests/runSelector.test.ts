// Run-selector view logic: label format, status dots, and the disabled states
// (mock, empty history, single bundle). Pure functions — no DOM required.
import { describe, expect, it } from "vitest"

import type { RunSummary } from "@/lib/tinct/dashboardData"
import {
  buildRunSelectorModel,
  formatRunDate,
  runDot,
  runOptionLabel,
  shortRunId,
} from "@/lib/tinct/runSelector"

function run(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: "cert_20260904_143022",
    timestamp: "2026-09-04T14:35:10+00:00",
    verdict: "SHIP",
    base_model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    trusted: true,
    integrity: "verified",
    ...overrides,
  }
}

describe("label formatting", () => {
  it("truncates long run ids and keeps short ones", () => {
    expect(shortRunId("cert_20260904_143022")).toBe("cert_…3022")
    expect(shortRunId("cert_x")).toBe("cert_x")
  })

  it("formats the date as month + day", () => {
    expect(formatRunDate("2026-09-04T14:35:10+00:00")).toBe("Sep 4")
    expect(formatRunDate("not-a-date")).toBe("")
  })

  it("builds `dot · id · verdict · date`", () => {
    expect(runOptionLabel(run())).toBe("🟢 cert_…3022 · SHIP · Sep 4")
    expect(runOptionLabel(run({ verdict: "DON'T SHIP" }))).toBe("🟢 cert_…3022 · DON'T SHIP · Sep 4")
  })
})

describe("status dots", () => {
  it("green for verified + trusted", () => {
    expect(runDot(run())).toBe("🟢")
  })

  it("amber when the math is valid but the issuer is unpinned", () => {
    expect(runDot(run({ trusted: false }))).toBe("🟡")
  })

  it("red wins over trust when integrity is not verified", () => {
    expect(runDot(run({ integrity: "tampered", trusted: true }))).toBe("🔴")
    expect(runDot(run({ integrity: "unsigned", trusted: false }))).toBe("🔴")
  })
})

describe("selector states", () => {
  it("renders disabled with a single mock entry in mock mode", () => {
    const model = buildRunSelectorModel({ source: "mock", currentRunId: "cert_x", runs: [] })
    expect(model.disabled).toBe(true)
    expect(model.options).toHaveLength(1)
    expect(model.options[0].label).toBe("🧪 mock")
    expect(model.value).toBe("mock")
  })

  it("renders disabled (not hidden) when the store is empty", () => {
    const model = buildRunSelectorModel({ source: "live", currentRunId: "cert_x", runs: [] })
    expect(model.disabled).toBe(true)
    expect(model.options).toHaveLength(1)
    expect(model.value).toBe("latest")
  })

  it("renders disabled with the single bundle so the affordance stays discoverable", () => {
    const only = run({ run_id: "cert_solo" })
    const model = buildRunSelectorModel({ source: "live", currentRunId: "cert_solo", runs: [only] })
    expect(model.disabled).toBe(true)
    expect(model.options.map((o) => o.value)).toEqual(["cert_solo"])
    expect(model.value).toBe("cert_solo")
  })

  it("enables with history and preserves descending order", () => {
    const newest = run({ run_id: "cert_new", timestamp: "2026-09-05T10:00:00+00:00" })
    const older = run({ run_id: "cert_old", timestamp: "2026-09-01T10:00:00+00:00" })
    const model = buildRunSelectorModel({
      source: "live",
      currentRunId: "cert_old",
      runs: [newest, older],
    })
    expect(model.disabled).toBe(false)
    expect(model.options.map((o) => o.value)).toEqual(["latest", "cert_new", "cert_old"])
    expect(model.value).toBe("cert_old")
  })

  it("falls back to latest when the viewed run is not in the list", () => {
    const model = buildRunSelectorModel({
      source: "live",
      currentRunId: "cert_not_listed",
      runs: [run({ run_id: "cert_a" }), run({ run_id: "cert_b" })],
    })
    expect(model.value).toBe("latest")
  })
})
