"use client"

// Vector PDF client report (@react-pdf/renderer). Imported dynamically from
// ExportPdfButton so the PDF engine never enters the server render path.
import { Document, Image, Page, Text, View, StyleSheet } from "@react-pdf/renderer"

import type { DashboardData } from "@/lib/tinct/dashboardData"
import { buildReportModel } from "@/lib/tinct/reportModel"

const styles = StyleSheet.create({
  page: { padding: 40, fontFamily: "Helvetica", fontSize: 10, color: "#18181b", lineHeight: 1.5 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 40, borderBottomWidth: 1, borderBottomColor: "#e4e4e7", paddingBottom: 10 },
  logo: { fontSize: 24, fontWeight: "bold", color: "#10b981" }, // Emerald
  title: { fontSize: 28, fontWeight: "bold", marginBottom: 8, color: "#09090b" },
  subtitle: { fontSize: 12, color: "#52525b", marginBottom: 30 },

  stampContainer: { alignItems: "center", marginVertical: 40 },
  stampPass: { width: 140, height: 140, borderRadius: 70, borderWidth: 6, borderColor: "#10b981", alignItems: "center", justifyContent: "center" },
  stampFail: { width: 140, height: 140, borderRadius: 70, borderWidth: 6, borderColor: "#ef4444", alignItems: "center", justifyContent: "center" },
  stampTextPass: { color: "#10b981", fontSize: 24, fontWeight: "bold" },
  stampTextFail: { color: "#ef4444", fontSize: 24, fontWeight: "bold" },

  metaGrid: { flexDirection: "row", flexWrap: "wrap", marginBottom: 30 },
  metaItem: { width: "50%", marginBottom: 10 },
  metaLabel: { fontSize: 9, color: "#71717a", textTransform: "uppercase", fontWeight: "bold" },
  metaValue: { fontSize: 11, color: "#18181b", fontFamily: "Courier" },

  warning: { backgroundColor: "#fef3c7", borderLeftWidth: 3, borderLeftColor: "#f59e0b", padding: 8, marginBottom: 8, fontSize: 9, color: "#78350f" },
  criticalWarning: { backgroundColor: "#fee2e2", borderLeftWidth: 3, borderLeftColor: "#ef4444", padding: 8, marginBottom: 8, fontSize: 9, color: "#7f1d1d" },

  sectionTitle: { fontSize: 16, fontWeight: "bold", marginBottom: 15, marginTop: 20, color: "#09090b", borderBottomWidth: 1, borderBottomColor: "#e4e4e7", paddingBottom: 5 },

  tableHeader: { flexDirection: "row", backgroundColor: "#f4f4f5", padding: 10, fontWeight: "bold", fontSize: 9, textTransform: "uppercase", color: "#3f3f46" },
  tableRow: { flexDirection: "row", padding: 10, borderBottomWidth: 1, borderBottomColor: "#e4e4e7" },
  colName: { width: "25%" },
  colStatus: { width: "15%", fontWeight: "bold" },
  colMetric: { width: "20%" },
  colDesc: { width: "40%", fontSize: 9, color: "#52525b" },
  rowReason: { fontSize: 8, color: "#b91c1c", marginTop: 2 },

  // Fixed box + objectFit: a capture with an unusual aspect ratio scales to
  // fit instead of overflowing the page (a 1:1 capture at width:100% would
  // otherwise add a page).
  chartImage: { width: "100%", height: 250, objectFit: "contain", marginTop: 10, marginBottom: 20, borderRadius: 4 },

  telemetryRow: { flexDirection: "row", paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: "#f4f4f5" },
  telemetryLabel: { width: "50%", color: "#52525b" },
  telemetryValue: { width: "50%", fontFamily: "Courier", fontWeight: "bold" },
  routeRow: { flexDirection: "row", paddingVertical: 5, borderBottomWidth: 1, borderBottomColor: "#f4f4f5", fontSize: 9 },
  routeCol: { width: "33%" },

  footer: { position: "absolute", bottom: 30, left: 40, right: 40, fontSize: 8, color: "#a1a1aa", textAlign: "center", borderTopWidth: 1, borderTopColor: "#e4e4e7", paddingTop: 10 },
})

export function ClientReportPDF({
  data,
  chartImage,
}: {
  data: DashboardData
  chartImage?: string | null
}) {
  const report = buildReportModel(data)
  const { isPass } = report

  return (
    <Document title={`tinct certification report — ${report.runId}`} author="tinct">
      {/* PAGE 1: Executive Summary & Stamp */}
      <Page size="A4" style={styles.page}>
        <View style={styles.header}>
          <Text style={styles.logo}>tinct</Text>
          <Text style={{ fontSize: 10, color: "#71717a" }}>Certification Report</Text>
        </View>

        <Text style={styles.title}>Model Safety Certification</Text>
        <Text style={styles.subtitle}>Forensic behavioral evaluation and cryptographic proof of safety.</Text>

        <View style={styles.stampContainer}>
          <View style={isPass ? styles.stampPass : styles.stampFail}>
            <Text style={isPass ? styles.stampTextPass : styles.stampTextFail}>{report.verdict}</Text>
          </View>
        </View>

        <View style={styles.metaGrid}>
          <View style={styles.metaItem}>
            <Text style={styles.metaLabel}>Run ID</Text>
            <Text style={styles.metaValue}>{report.runId}</Text>
          </View>
          <View style={styles.metaItem}>
            <Text style={styles.metaLabel}>Base Model</Text>
            <Text style={styles.metaValue}>{report.baseModel}</Text>
          </View>
          <View style={styles.metaItem}>
            <Text style={styles.metaLabel}>Adapter / Weights</Text>
            <Text style={styles.metaValue}>{report.adapter}</Text>
          </View>
          <View style={styles.metaItem}>
            <Text style={styles.metaLabel}>Completed</Text>
            <Text style={styles.metaValue}>{report.completed}</Text>
          </View>
          <View style={styles.metaItem}>
            <Text style={styles.metaLabel}>Training Tool</Text>
            <Text style={styles.metaValue}>{data.trainingTool}</Text>
          </View>
          <View style={styles.metaItem}>
            <Text style={styles.metaLabel}>Evidence</Text>
            <Text style={styles.metaValue}>{report.trustLine}</Text>
          </View>
        </View>

        {report.warnings.map((line) => (
          <Text
            key={line}
            style={line.startsWith("MOCK") || line.startsWith("UNTRUSTED") ? styles.criticalWarning : styles.warning}
          >
            {line}
          </Text>
        ))}

        <Text style={{ fontSize: 11, marginTop: 20 }}>{report.summary}</Text>

        <Text style={styles.footer}>
          Generated by tinct | Open Source ML Safety Engine | Cryptographically Signed Evidence Bundle Attached
        </Text>
      </Page>

      {/* PAGE 2: The 6 Safety Gates */}
      <Page size="A4" style={styles.page}>
        <Text style={styles.sectionTitle}>Safety Gate Results</Text>

        {report.failedGateCount > 0 && (
          <Text style={styles.criticalWarning}>
            Certification failed on {report.failedGateCount} of {report.gates.length} gates. {report.rootCause}
          </Text>
        )}

        <View style={styles.tableHeader}>
          <Text style={styles.colName}>Gate</Text>
          <Text style={styles.colStatus}>Status</Text>
          <Text style={styles.colMetric}>Key Metric</Text>
          <Text style={styles.colDesc}>Description</Text>
        </View>

        {report.gates.map((gate) => (
          <View style={styles.tableRow} key={gate.key} wrap={false}>
            <View style={styles.colName}>
              <Text>{gate.title}</Text>
            </View>
            <Text style={[styles.colStatus, { color: gate.status === "PASS" ? "#10b981" : gate.status === "FAIL" ? "#ef4444" : "#71717a" }]}>
              {gate.status}
            </Text>
            <Text style={styles.colMetric}>{gate.metric}</Text>
            <View style={styles.colDesc}>
              <Text>{gate.description}</Text>
              {gate.reason ? <Text style={styles.rowReason}>{gate.reason}</Text> : null}
            </View>
          </View>
        ))}

        <Text style={styles.footer}>Generated by tinct | Evidence Hash: {report.runId}</Text>
      </Page>

      {/* PAGE 3: Hardware Telemetry — live chart snapshot (when captured),
          with the numeric tables underneath for the auditors. */}
      <Page size="A4" style={styles.page}>
        <Text style={styles.sectionTitle}>Hardware Telemetry</Text>
        {chartImage ? (
          <Image src={chartImage} style={styles.chartImage} />
        ) : (
          <Text style={{ fontSize: 9, color: "#71717a", marginBottom: 10 }}>
            No chart snapshot was captured for this run; the numeric tables below are authoritative.
          </Text>
        )}
        <Text style={{ ...styles.sectionTitle, marginTop: 10 }}>MoEStreamer Telemetry</Text>
        {report.telemetry.map((row) => (
          <View style={styles.telemetryRow} key={row.label}>
            <Text style={styles.telemetryLabel}>{row.label}</Text>
            <Text style={styles.telemetryValue}>{row.value}</Text>
          </View>
        ))}

        <Text style={styles.sectionTitle}>Expert Routing (Base vs Adapter)</Text>
        {report.routing.length > 0 ? (
          <>
            <View style={styles.tableHeader}>
              <Text style={styles.routeCol}>Expert</Text>
              <Text style={styles.routeCol}>Base</Text>
              <Text style={styles.routeCol}>Adapter</Text>
            </View>
            {report.routing.map((row) => (
              <View style={styles.routeRow} key={row.expert}>
                <Text style={styles.routeCol}>{row.expert}</Text>
                <Text style={styles.routeCol}>{row.base}</Text>
                <Text style={[styles.routeCol, { color: row.starved ? "#ef4444" : "#18181b", fontWeight: row.starved ? "bold" : "normal" }]}>
                  {row.adapter}
                  {row.starved ? "  (starved)" : ""}
                </Text>
              </View>
            ))}
          </>
        ) : (
          <Text style={{ fontSize: 10, color: "#52525b" }}>
            No MoE routing data in this bundle (dense-model run).
          </Text>
        )}

        <Text style={styles.footer}>
          Generated by tinct | Verify this bundle with `tinct security check --run {report.runId}`
        </Text>
      </Page>
    </Document>
  )
}
