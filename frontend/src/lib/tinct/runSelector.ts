// Run-selector view logic, kept pure so it can be unit-tested without a DOM.
//
// Label format:  <dot> cert_…3022 · SHIP · Sep 4
// Dots encode the two independent claims, integrity first (it is the stronger
// statement): red = integrity !== "verified", amber = verified but the issuer
// key is not pinned, green = verified + trusted.
import type { RunSummary } from "@/lib/tinct/dashboardData"

export interface RunOption {
  value: string
  label: string
}

export interface RunSelectorModel {
  disabled: boolean
  options: RunOption[]
  value: string
}

const DOT_GREEN = "🟢"
const DOT_AMBER = "🟡"
const DOT_RED = "🔴"

/** `cert_20260904_143022` → `cert_…3022` */
export function shortRunId(runId: string): string {
  if (runId.length <= 12) return runId
  return `${runId.slice(0, 5)}…${runId.slice(-4)}`
}

export function runDot(run: RunSummary): string {
  if (run.integrity !== "verified") return DOT_RED
  return run.trusted ? DOT_GREEN : DOT_AMBER
}

/** ISO timestamp → "Sep 4" (empty string when unparseable). */
export function formatRunDate(timestamp: string): string {
  const parsed = Date.parse(timestamp)
  if (Number.isNaN(parsed)) return ""
  return new Date(parsed).toLocaleDateString("en-US", { month: "short", day: "numeric" })
}

export function runOptionLabel(run: RunSummary): string {
  const date = formatRunDate(run.timestamp)
  const parts = [shortRunId(run.run_id), run.verdict, date].filter(Boolean)
  return `${runDot(run)} ${parts.join(" · ")}`
}

/**
 * The selector always renders (the affordance must be discoverable before CI
 * volume arrives) — it is disabled when there is nothing to choose between.
 */
export function buildRunSelectorModel(params: {
  source: "live" | "mock"
  currentRunId: string
  runs: RunSummary[]
}): RunSelectorModel {
  const { source, currentRunId, runs } = params

  if (source === "mock") {
    return { disabled: true, options: [{ value: "mock", label: "🧪 mock" }], value: "mock" }
  }

  if (runs.length === 0) {
    return { disabled: true, options: [{ value: "latest", label: "no runs yet" }], value: "latest" }
  }

  if (runs.length === 1) {
    // One bundle: still show it, but disabled — history will fill in.
    const only = runs[0]
    return { disabled: true, options: [{ value: only.run_id, label: runOptionLabel(only) }], value: only.run_id }
  }

  const options: RunOption[] = [
    { value: "latest", label: "latest" },
    ...runs.map((run) => ({ value: run.run_id, label: runOptionLabel(run) })),
  ]
  const value = runs.some((run) => run.run_id === currentRunId) ? currentRunId : "latest"
  return { disabled: false, options, value }
}
