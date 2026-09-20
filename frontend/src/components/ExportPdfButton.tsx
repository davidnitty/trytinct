"use client"

// Exports the current run as a vector PDF client report.
// The PDF engine is imported on click so it never enters the server render
// path (react-pdf is browser-oriented), and a slow first render only affects
// the click that asked for it.
import { useState } from "react"
import { FileText } from "lucide-react"

import { buttonVariants } from "@/components/ui/button"
import type { DashboardData } from "@/lib/tinct/dashboardData"

export function ExportPdfButton({ data }: { data: DashboardData }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function exportPdf() {
    setBusy(true)
    setError(null)
    try {
      const [{ pdf }, { ClientReportPDF }] = await Promise.all([
        import("@react-pdf/renderer"),
        import("@/components/ClientReportPDF"),
      ])
      const blob = await pdf(<ClientReportPDF data={data} />).toBlob()

      const url = URL.createObjectURL(blob)
      const anchor = document.createElement("a")
      anchor.href = url
      anchor.download = `tinct_report_${data.runId}.pdf`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "PDF generation failed")
    } finally {
      setBusy(false)
    }
  }

  return (
    <span className="inline-flex items-center gap-3">
      <button
        type="button"
        onClick={exportPdf}
        disabled={busy}
        className={
          buttonVariants({ variant: "outline" }) +
          " border-white/20 bg-transparent text-white hover:bg-white/10 gap-2 disabled:opacity-60"
        }
      >
        <FileText className="w-4 h-4" />
        {busy ? "Generating…" : "Export Client Report (PDF)"}
      </button>
      {error && <span className="text-xs text-red-400">{error}</span>}
    </span>
  )
}
