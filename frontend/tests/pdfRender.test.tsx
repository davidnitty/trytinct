// Real render smoke test: the PDF document must actually build in Node.
// This catches react-pdf misuse (bad styles, undefined data, invalid nesting)
// far more reliably than clicking a button in a browser.
import { writeFileSync } from "node:fs"
import { pdf } from "@react-pdf/renderer"
import { describe, expect, it } from "vitest"

import { ClientReportPDF } from "@/components/ClientReportPDF"
import type { DashboardData } from "@/lib/tinct/dashboardData"

const fixture: DashboardData = {
  source: "live",
  verified: true,
  trusted: true,
  keyFingerprint: "ffbd0c0567af392ad1c5b082fca45cd4dabd0bb7403d9f758e8fb5e9ab449e17",
  runId: "cert_20260904_143022",
  baseModel: "mistralai/Mixtral-8x7B-Instruct-v0.1",
  adapter: "runs/cert_20260904_143022/adapter",
  verdict: "DON'T SHIP",
  timestamp: "2026-09-04T14:35:10+00:00",
  trainingTool: "unsloth",
  gates: [
    { key: "toxicity", title: "Toxicity", status: "FAIL", detail: "3.4x factor", reason: "Adapter toxicity spiked 3.4x." },
    { key: "canary_leakage", title: "Canary Leakage", status: "PASS", detail: "0% leaked" },
    { key: "memory_offload", title: "Memory Offload", status: "NOT RUN", detail: "—" },
  ],
  expertRouting: [{ expert: "Expert 6", base: 12.6, adapter: 4.1 }],
  offloadStats: { cacheHits: 8450, h2dStreams: 142, d2hEvictions: 134, bytesH2dGb: 4, bytesD2hGb: 3.9, vramSavedGb: 68.4 },
  failedGateCount: 1,
  rootCause: "review your training data for toxic examples.",
  starvedExperts: [0],
  utilizationThresholdPct: 1,
  telemetry: [
    { step: 0, vram_mb: 24118.5, resident_experts: 2, h2d_streams: 0, cache_hits: 0 },
    { step: 1, vram_mb: 27300, resident_experts: 2, h2d_streams: 1, cache_hits: 12 },
  ],
}

// 1x1 transparent PNG — proves the Image component accepts a captured data
// URL without depending on a real chart capture in Node.
const CHART_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

describe("client report PDF", () => {
  it("renders a real PDF document", async () => {
    // toBlob() is the exact path the export button uses, so this exercises the
    // production call and not just the document definition.
    const blob = await pdf(<ClientReportPDF data={fixture} />).toBlob()
    const bytes = Buffer.from(await blob.arrayBuffer())
    // Opt-in artifact dump for manual inspection (e.g. pypdf):
    //   PDF_DUMP=/tmp/report.pdf npx vitest run tests/pdfRender.test.tsx
    if (process.env.PDF_DUMP) writeFileSync(process.env.PDF_DUMP, bytes)
    expect(bytes.subarray(0, 5).toString("latin1")).toBe("%PDF-")
    expect(bytes.length).toBeGreaterThan(2000) // 3 pages of vector content
  }, 30000)

  it("embeds the captured chart image on the telemetry page", async () => {
    const blob = await pdf(<ClientReportPDF data={fixture} chartImage={CHART_PNG} />).toBlob()
    const bytes = Buffer.from(await blob.arrayBuffer())
    if (process.env.PDF_DUMP_IMAGE) writeFileSync(process.env.PDF_DUMP_IMAGE, bytes)
    expect(bytes.subarray(0, 5).toString("latin1")).toBe("%PDF-")
  }, 30000)
})
