"use client"

import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  ShieldCheck, Ghost, MessageSquareOff, Biohazard, BrainCircuit, HardDrive,
  ArrowLeft, CheckCircle2, XCircle, Cpu, Zap
} from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { mockRunData } from "@/lib/mockData"

// Helper to format bytes to GB
const formatGB = (bytes: number) => `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`

export default function DashboardPage() {
  const { runId, baseModel, adapter, verdict, timestamp, gates, expertRouting, offloadStats } = mockRunData
  const isPass = verdict === "SHIP"

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
          <Badge variant="outline" className="h-6 border-white/20 bg-white/5 font-mono text-xs text-gray-300">{runId}</Badge>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-12 space-y-12">

        {/* 1. Verdict Header */}
        <section className="grid lg:grid-cols-3 gap-8 items-center">
          <div className="lg:col-span-2 space-y-4">
            <div className="flex items-center gap-3">
              <h1 className="text-4xl font-bold tracking-tight">Certification Report</h1>
              <Badge variant={isPass ? "default" : "destructive"} className={`text-lg px-4 py-1 ${isPass ? 'bg-emerald-500 text-black hover:bg-emerald-600' : ''}`}>
                {verdict}
              </Badge>
            </div>
            <p className="text-gray-400 text-lg">
              Evaluated <span className="text-white font-mono">{adapter}</span> against base model <span className="text-white font-mono">{baseModel}</span>.
            </p>
            <p className="text-sm text-gray-500" suppressHydrationWarning>Completed {new Date(timestamp).toLocaleString()}</p>
          </div>

          {/* Giant Stamp */}
          <div className="flex justify-center lg:justify-end">
            <div className={`relative w-48 h-48 rounded-full border-8 flex items-center justify-center transform -rotate-12 ${isPass ? 'border-emerald-500/50' : 'border-red-500/50'}`}>
               <div className={`text-center ${isPass ? 'text-emerald-500' : 'text-red-500'}`}>
                  {isPass ? <CheckCircle2 className="w-16 h-16 mx-auto mb-2" /> : <XCircle className="w-16 h-16 mx-auto mb-2" />}
                  <p className="text-2xl font-black tracking-widest">{verdict}</p>
               </div>
            </div>
          </div>
        </section>

        {/* 2. The 6 Gates Grid */}
        <section>
          <h2 className="text-2xl font-bold mb-6">Safety Gates</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <GateCard icon={Ghost} title="Canary Leakage" status={gates.canary_leakage.status} metric={`${gates.canary_leakage.leakage_rate}% leaked`} />
            <GateCard icon={MessageSquareOff} title="Refusal Regression" status={gates.refusal_regression.status} metric={`${(gates.refusal_regression.regression_rate * 100).toFixed(1)}% delta`} />
            <GateCard icon={Biohazard} title="Toxicity" status={gates.toxicity.status} metric={`${gates.toxicity.increase_factor}x factor`} />
            <GateCard icon={BrainCircuit} title="Expert Collapse" status={gates.expert_collapse.status} metric={`Min util: ${(gates.expert_collapse.min_utilization * 100).toFixed(1)}%`} />
            <GateCard icon={ShieldCheck} title="Routing Regression" status={gates.routing_regression.status} metric={`${gates.routing_regression.regressed_experts.length} regressed`} />
            <GateCard icon={HardDrive} title="Memory Offload" status={gates.memory_offload.status} metric={`${gates.memory_offload.cache_hits} cache hits`} />
          </div>
        </section>

        {/* 3. Deep Dive Charts */}
        <section className="grid lg:grid-cols-2 gap-6">
          {/* Expert Routing Chart */}
          <Card className="bg-zinc-900/50 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <BrainCircuit className="w-5 h-5 text-emerald-400" /> MoE Expert Routing
              </CardTitle>
              <CardDescription>Base vs Adapter traffic distribution across 8 experts.</CardDescription>
            </CardHeader>
            <CardContent className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={expertRouting}>
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
            </CardContent>
          </Card>

          {/* MoE Streamer Stats */}
          <Card className="bg-zinc-900/50 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Cpu className="w-5 h-5 text-emerald-400" /> MoEStreamer Telemetry
              </CardTitle>
              <CardDescription>Hardware utilization during certification.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-6">
              <StatBox label="VRAM Saved" value={`${offloadStats.vram_saved_gb} GB`} icon={<Zap className="w-4 h-4 text-yellow-500" />} />
              <StatBox label="Resident Slots" value={`${offloadStats.resident_slots} / ${offloadStats.total_experts}`} icon={<HardDrive className="w-4 h-4 text-blue-500" />} />
              <StatBox label="H2D Streams" value={formatGB(offloadStats.bytes_h2d)} icon={<ArrowLeft className="w-4 h-4 text-emerald-500 rotate-180" />} />
              <StatBox label="D2H Evictions" value={formatGB(offloadStats.bytes_d2h)} icon={<ArrowLeft className="w-4 h-4 text-red-500" />} />
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
              <p className="text-sm text-gray-400 mt-1">This report is signed with Ed25519. Tamper-proof and verifiable.</p>
            </div>
            <Button variant="outline" className="border-white/20 bg-transparent text-white hover:bg-white/10">
              Download evidence.json
            </Button>
          </div>
        </section>

      </div>
    </main>
  )
}

// --- Subcomponents ---

function GateCard({ icon: Icon, title, status, metric }: { icon: any, title: string, status: string, metric: string }) {
  const isPass = status === "PASS"
  return (
    <Card className={`bg-zinc-900/50 border-white/10 ${isPass ? 'hover:border-emerald-500/30' : 'hover:border-red-500/30'} transition-colors`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <Icon className={`w-5 h-5 ${isPass ? 'text-emerald-400' : 'text-red-500'}`} />
          <Badge variant={isPass ? "outline" : "destructive"} className={`${isPass ? 'text-emerald-400 border-emerald-400/50' : ''} text-xs`}>
            {status}
          </Badge>
        </div>
        <CardTitle className="text-white text-lg mt-2">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-bold text-white font-mono">{metric}</p>
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