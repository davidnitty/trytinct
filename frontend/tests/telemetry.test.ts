// Telemetry mapping: the signed per-step series must reach the view model
// (the chart reads data.telemetry), and malformed entries must not produce
// NaN — the series is signed data, not trusted code.
import { describe, expect, it } from "vitest"

import {
  bundleToDashboardData,
  mockToDashboardData,
  telemetryPoints,
} from "@/lib/tinct/dashboardData"

/* eslint-disable @typescript-eslint/no-explicit-any */

function bundleWith(offloadTelemetry: unknown): Record<string, any> {
  return {
    project_name: "fixture",
    model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    family: "mistral",
    decision: "SHIP",
    created_at: "2026-09-04T14:35:10+00:00",
    artifacts: {},
    config: {},
    training_tool: "unsloth",
    training_executed: false,
    safety_gates: {
      result: "PASS",
      offload_stats: { h2d_streams: 2, d2h_evictions: 1, cache_hits: 9, bytes_h2d: 1024, bytes_d2h: 512 },
      offload_telemetry: offloadTelemetry,
    },
  }
}

const VERIFICATION = { signatureValid: true, trusted: true, keyFingerprint: "ff".repeat(32) }

describe("telemetryPoints", () => {
  it("maps well-formed snapshots", () => {
    const points = telemetryPoints([
      { step: 0, vram_mb: 24118.5, resident_experts: 2, h2d_streams: 0, cache_hits: 0 },
      { step: 1, vram_mb: 26700, resident_experts: 2, h2d_streams: 1, cache_hits: 12 },
    ])
    expect(points).toHaveLength(2)
    expect(points[1]).toEqual({
      step: 1,
      vram_mb: 26700,
      resident_experts: 2,
      h2d_streams: 1,
      cache_hits: 12,
    })
  })

  it("returns an empty series for missing or non-array data", () => {
    expect(telemetryPoints(undefined)).toEqual([])
    expect(telemetryPoints(null)).toEqual([])
    expect(telemetryPoints("nope")).toEqual([])
    expect(telemetryPoints({})).toEqual([])
  })

  it("coerces junk entries instead of emitting NaN", () => {
    const points = telemetryPoints([
      { step: "3", vram_mb: "123.5", cache_hits: null },
      "garbage",
      { vram_mb: 10 },
    ])
    expect(points).toHaveLength(2) // the string entry is dropped
    for (const point of points) {
      for (const value of Object.values(point)) {
        expect(Number.isNaN(value)).toBe(false)
      }
    }
    expect(points[0]).toEqual({ step: 3, vram_mb: 123.5, resident_experts: 0, h2d_streams: 0, cache_hits: 0 })
    expect(points[1].step).toBe(1) // falls back to the array index
  })
})

describe("view model wiring", () => {
  it("surfaces signed telemetry from the bundle", () => {
    const data = bundleToDashboardData(
      "cert_x",
      bundleWith([{ step: 0, vram_mb: 24118.5, resident_experts: 2, h2d_streams: 0, cache_hits: 0 }]),
      VERIFICATION,
    )
    expect(data.telemetry).toHaveLength(1)
    expect(data.telemetry[0].vram_mb).toBe(24118.5)
  })

  it("yields an empty series for bundles without telemetry (pre-Phase-2 or dense runs)", () => {
    const data = bundleToDashboardData("cert_old", bundleWith(undefined), VERIFICATION)
    expect(data.telemetry).toEqual([])
  })

  it("gives the mock scenarios a series so the demo chart has data", () => {
    expect(mockToDashboardData("pass").telemetry.length).toBeGreaterThan(0)
    expect(mockToDashboardData("fail").telemetry.length).toBeGreaterThan(0)
  })
})
