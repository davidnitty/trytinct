"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { buttonVariants } from "@/components/ui/button"
import {
  ShieldCheck, Ghost, MessageSquareOff, Biohazard, BrainCircuit, HardDrive,
  ArrowLeft, CheckCircle2, XCircle, Cpu, Zap, FolderOpen, ShieldAlert, FlaskConical,
} from "lucide-react"
import type { LucideIcon } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import type { DashboardData, GateView } from "@/lib/tinct/dashboardData"

const GATE_ICONS: Record<string, LucideIcon> = {
  canary_leakage: Ghost,
  refusal_regression: MessageSquareOff,
  toxicity: Biohazard,
  expert_collapse: BrainCircuit,
  routing_regression: ShieldCheck,
  memory_offload: HardDrive,
}

type LoadState =
  | { status: "loading" }
  | { status: "empty"; message: string }
  | { status: "invalid"; message: string }
  | { status: "ready"; data: DashboardData }

export default function DashboardPage() {
  const [state, setState] = useState<LoadState>({ status: "loading" })

  useEffect(() => {
    const mock = new URLSearchParams(window.location.search).get("mock") === "1"
    fetch(`/api/evidence/latest${mock ? "?mock=1" : ""}`)
      .then(async (res) => {
        const body = await res.json()
        if (res.ok) {
          setState({ status: "ready", data: body.data })
        } else if (res.status === 404) {
          setState({ status: "empty", message: body.message })
        } else {
          setState({ status: "invalid", message: body.message })
        }
      })
      .catch(() => setState({ status: "invalid", message: "Could not reach the evidence API." }))
  }, [])

  return (
    <main className="min-h-screen bg-black text-white antialiased">
      {/* Top Nav */}
      <nav className="sticky top-0 z-50 border-b border-white/10 bg-black/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-white transition">
              <ArrowLeft className="w-4 h-4" /> Back to Home
            </Link>
            <div className="h-6 w-px bg-white/10"></div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 bg-emerald-500 rounded flex items-center justify-center font-bold text-black text-xs">t</div>
              <span className="font-semibold tracking-tight">tinct dashboard</span>
            </div>
          </div>
          {state.status === "ready" ? (
            <Badge variant="outline" className="h-6 border-white/20 bg-white/5 font-mono text-xs text-gray-300">
              {state.data.runId}
            </Badge>
          ) : (
            <div className="h-5 w-40 animate-pulse rounded-full bg-white/10" />
          )}
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-12">
        {state.status === "loading" && <DashboardSkeleton />}
        {state.status === "empty" && <EmptyState message={state.message} />}
        {state.status === "invalid" && <InvalidState message={state.message} />}
        {state.status === "ready" && <Report data={state.data} />}
      </div>
    </main>
  )
}

/* ------------------------------------------------------------------ states */

function DashboardSkeleton() {
  return (
    <div className="space-y-12" aria-busy="true" aria-label="Loading certification report">
      <section className="grid lg:grid-cols-3 gap-8 items-center">
        <div className="lg:col-span-2 space-y-4">
          <div className="h-10 w-96 animate-pulse rounded-lg bg-white/10" />
          <div className="h-6 w-[28rem] max-w-full animate-pulse rounded bg-white/5" />
          <div className="h-4 w-56 animate-pulse rounded bg-white/5" />
        </div>
        <div className="flex justify-center lg:justify-end">
          <div className="w-48 h-48 rounded-full border-8 border-white/10 animate-pulse" />
        </div>
      </section>

      <section>
        <div className="h-7 w-40 animate-pulse rounded bg-white/10 mb-6" />
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="bg-zinc-900/50 border-white/10">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="h-5 w-5 animate-pulse rounded bg-white/10" />
                  <div className="h-5 w-14 animate-pulse rounded-full bg-white/10" />
                </div>
                <div className="h-5 w-36 animate-pulse rounded bg-white/5 mt-2" />
              </CardHeader>
              <CardContent>
                <div className="h-8 w-32 animate-pulse rounded bg-white/10" />
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        <Card className="bg-zinc-900/50 border-white/10">
          <CardHeader>
            <div className="h-6 w-56 animate-pulse rounded bg-white/10" />
            <div className="h-4 w-72 animate-pulse rounded bg-white/5" />
          </CardHeader>
          <CardContent className="h-80">
            <div className="h-full w-full animate-pulse rounded-lg bg-white/5" />
          </CardContent>
        </Card>
        <Card className="bg-zinc-900/50 border-white/10">
          <CardHeader>
            <div className="h-6 w-56 animate-pulse rounded bg-white/10" />
            <div className="h-4 w-64 animate-pulse rounded bg-white/5" />
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-6">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-24 animate-pulse rounded-lg bg-white/5" />
            ))}
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-32 text-center space-y-6">
      <FolderOpen className="w-16 h-16 text-gray-600" />
      <div className="space-y-2">
        <h1 className="text-2xl font-bold">No evidence bundle found</h1>
        <p className="text-gray-400 max-w-lg">{message}</p>
      </div>
      <Link
        href="/dashboard?mock=1"
        className={buttonVariants({ variant: "outline" }) + " border-white/20 bg-transparent text-white hover:bg-white/10 gap-2"}
      >
        <FlaskConical className="w-4 h-4" /> View demo data instead
      </Link>
    </div>
  )
}

function InvalidState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-32 text-center space-y-6">
      <ShieldAlert className="w-16 h-16 text-red-500" />
      <div className="space-y-2">
        <h1 className="text-2xl font-bold text-red-400">Evidence verification failed</h1>
        <p className="text-gray-400 max-w-lg">{message}</p>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ report */

function Report({ data }: { data: DashboardData }) {
  const isPass = data.verdict === "SHIP"
  return (
    <div className="space-y-12">
      {/* 1. Verdict Header */}
      <section className="grid lg:grid-cols-3 gap-8 items-center">
        <div className="lg:col-span-2 space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-4xl font-bold tracking-tight">Certification Report</h1>
            <Badge variant={isPass ? "default" : "destructive"} className={`text-lg px-4 py-1 ${isPass ? "bg-emerald-500 text-black hover:bg-emerald-600" : ""}`}>
              {data.verdict}
            </Badge>
            {data.source === "mock" ? (
              <Badge variant="outline" className="gap-1 border-amber-400/50 bg-amber-400/10 text-amber-300">
                <FlaskConical className="w-3 h-3" /> MOCK DATA
              </Badge>
            ) : (
              data.verified && (
                <Badge variant="outline" className="gap-1 border-emerald-400/50 bg-emerald-400/10 text-emerald-300">
                  <ShieldCheck className="w-3 h-3" /> Ed25519 VERIFIED
                </Badge>
              )
            )}
          </div>
          <p className="text-gray-400 text-lg">
            Evaluated <span className="text-white font-mono">{data.adapter}</span> against base model{" "}
            <span className="text-white font-mono">{data.baseModel}</span>.
          </p>
          <p className="text-sm text-gray-500" suppressHydrationWarning>
            Completed {data.timestamp ? new Date(data.timestamp).toLocaleString() : "—"}
            {data.source === "live" ? ` · trained with ${data.trainingTool}` : ""}
          </p>
        </div>

        {/* Giant Stamp */}
        <div className="flex justify-center lg:justify-end">
          <div className={`relative w-48 h-48 rounded-full border-8 flex items-center justify-center transform -rotate-12 ${isPass ? "border-emerald-500/50" : "border-red-500/50"}`}>
            <div className={`text-center ${isPass ? "text-emerald-500" : "text-red-500"}`}>
              {isPass ? <CheckCircle2 className="w-16 h-16 mx-auto mb-2" /> : <XCircle className="w-16 h-16 mx-auto mb-2" />}
              <p className="text-2xl font-black tracking-widest">{data.verdict}</p>
            </div>
          </div>
        </div>
      </section>

      {/* 2. The 6 Gates Grid */}
      <section>
        <h2 className="text-2xl font-bold mb-6">Safety Gates</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.gates.map((gate) => (
            <GateCard key={gate.key} gate={gate} />
          ))}
        </div>
      </section>

      {/* 3. Deep Dive Charts */}
      <section className="grid lg:grid-cols-2 gap-6">
        <Card className="bg-zinc-900/50 border-white/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <BrainCircuit className="w-5 h-5 text-emerald-400" /> MoE Expert Routing
            </CardTitle>
            <CardDescription>Base vs Adapter traffic distribution per expert.</CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            {data.expertRouting.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.expertRouting}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="expert" stroke="#71717a" fontSize={12} />
                  <YAxis stroke="#71717a" fontSize={12} unit="%" />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "8px" }}
                    labelStyle={{ color: "#fff" }}
                  />
                  <Legend />
                  <Bar dataKey="base" fill="#71717a" name="Base Model" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="adapter" fill="#10b981" name="Adapter" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500 text-sm">
                No MoE routing data in this bundle (dense-model run).
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="bg-zinc-900/50 border-white/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Cpu className="w-5 h-5 text-emerald-400" /> MoEStreamer Telemetry
            </CardTitle>
            <CardDescription>Hardware utilization during certification.</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-6">
            {data.offloadStats.vramSavedGb !== null && (
              <StatBox label="VRAM Saved" value={`${data.offloadStats.vramSavedGb} GB`} icon={<Zap className="w-4 h-4 text-yellow-500" />} />
            )}
            <StatBox label="Cache Hits" value={data.offloadStats.cacheHits.toLocaleString()} icon={<Zap className="w-4 h-4 text-emerald-400" />} />
            <StatBox label="H2D Streams" value={String(data.offloadStats.h2dStreams)} icon={<ArrowLeft className="w-4 h-4 text-emerald-500 rotate-180" />} />
            <StatBox label="D2H Evictions" value={String(data.offloadStats.d2hEvictions)} icon={<ArrowLeft className="w-4 h-4 text-red-500" />} />
            <StatBox label="H2D Transferred" value={`${data.offloadStats.bytesH2dGb} GB`} icon={<ArrowLeft className="w-4 h-4 text-emerald-500 rotate-180" />} />
            <StatBox label="D2H Transferred" value={`${data.offloadStats.bytesD2hGb} GB`} icon={<ArrowLeft className="w-4 h-4 text-red-500" />} />
          </CardContent>
        </Card>
      </section>

      {/* 4. Evidence Signature */}
      <section className="border-t border-white/10 pt-8">
        <div className="flex items-center justify-between bg-zinc-900/30 border border-white/5 rounded-lg p-6">
          <div>
            <h3 className="font-semibold text-white flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" /> Cryptographic Evidence
            </h3>
            <p className="text-sm text-gray-400 mt-1">
              {data.source === "live"
                ? "This report is signed with Ed25519 and verified server-side. Tamper-proof and verifiable."
                : "This is mock demo data — no signed bundle exists on disk."}
            </p>
          </div>
          {data.source === "live" ? (
            <a
              href="/api/evidence"
              className={buttonVariants({ variant: "outline" }) + " border-white/20 bg-transparent text-white hover:bg-white/10"}
            >
              Download evidence.json
            </a>
          ) : (
            <span className="font-mono text-xs text-gray-500">no bundle on disk</span>
          )}
        </div>
      </section>
    </div>
  )
}

/* ------------------------------------------------------------ subcomponents */

function GateCard({ gate }: { gate: GateView }) {
  const Icon = GATE_ICONS[gate.key] ?? ShieldCheck
  const tone = gate.status === "PASS" ? "emerald" : gate.status === "FAIL" ? "red" : "gray"
  const iconClass =
    tone === "emerald" ? "text-emerald-400" : tone === "red" ? "text-red-500" : "text-gray-500"
  return (
    <Card className={`bg-zinc-900/50 border-white/10 ${tone === "emerald" ? "hover:border-emerald-500/30" : tone === "red" ? "hover:border-red-500/30" : ""} transition-colors`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <Icon className={`w-5 h-5 ${iconClass}`} />
          <Badge
            variant={gate.status === "FAIL" ? "destructive" : "outline"}
            className={`text-xs ${tone === "emerald" ? "text-emerald-400 border-emerald-400/50" : tone === "gray" ? "text-gray-400 border-white/20" : ""}`}
          >
            {gate.status}
          </Badge>
        </div>
        <CardTitle className="text-white text-lg mt-2">{gate.title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-bold text-white font-mono">{gate.detail}</p>
      </CardContent>
    </Card>
  )
}

function StatBox({ label, value, icon }: { label: string, value: string, icon: React.ReactNode }) {
  return (
    <div className="space-y-2 bg-black/20 p-4 rounded-lg border border-white/5">
      <div className="flex items-center gap-2 text-gray-400 text-sm">
        {icon} {label}
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
    </div>
  )
}
