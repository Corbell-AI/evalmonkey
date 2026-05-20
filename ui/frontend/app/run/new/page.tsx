'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { api } from '@/lib/api'
import { BenchmarkInfo, CATEGORY_COLORS } from '@/lib/types'
import { CHAOS_PROFILES, CODING_CHAOS_PROFILES, EVAL_MODELS } from '@/lib/benchmarks'
import { ChevronRight, Zap, Bot, FlaskConical } from 'lucide-react'

type Step = 1 | 2 | 3

export default function NewRunPage() {
  const router = useRouter()
  const [step, setStep] = useState<Step>(1)
  const [benchmarks, setBenchmarks] = useState<BenchmarkInfo[]>([])
  const [loading, setLoading] = useState(false)

  const [targetUrl, setTargetUrl] = useState('http://localhost:8000')
  const [useSampleAgent, setUseSampleAgent] = useState(false)
  const [requestKey, setRequestKey] = useState('question')
  const [responsePath, setResponsePath] = useState('data')
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string>('All')
  const [evalModel, setEvalModel] = useState('gpt-4o')
  const [limit, setLimit] = useState(5)
  const [enableChaos, setEnableChaos] = useState(false)
  const [chaosProfile, setChaosProfile] = useState('client_prompt_injection')
  const [showCodingChaosOnly, setShowCodingChaosOnly] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Auto-select the judge model matching whatever is configured in .env
    api.getConfig().then(cfg => setEvalModel(cfg.default_eval_model)).catch(() => {})
    api.listBenchmarks().then(setBenchmarks).catch(() => {})
  }, [])

  const categories = Array.from(new Set(benchmarks.map(b => b.category)))

  // Detect if the selected benchmark is a coding-category one
  const selectedBenchmarkInfo = benchmarks.find(b => b.id === selectedBenchmark)
  const isCodingBenchmark = selectedBenchmarkInfo?.category === 'Coding'

  // Visible chaos profiles: auto-surface coding ones when benchmark is Coding
  const visibleChaosProfiles = showCodingChaosOnly
    ? CODING_CHAOS_PROFILES
    : isCodingBenchmark && !showCodingChaosOnly
    ? CHAOS_PROFILES  // show all but coding ones bubble first
    : CHAOS_PROFILES.filter(p => p.category === 'general')

  const handleLaunch = async () => {
    if (!selectedBenchmark) return
    setLoading(true); setError(null)
    try {
      const base = {
        scenario: selectedBenchmark,
        target_url: useSampleAgent ? 'http://127.0.0.1:8001/solve' : targetUrl,
        eval_model: evalModel, request_key: requestKey,
        response_path: responsePath, limit, use_sample_agent: useSampleAgent,
      }
      const result = enableChaos
        ? await api.startChaos({ ...base, chaos_profile: chaosProfile })
        : await api.startBenchmark(base)
      router.push(`/run/${result.run_id}`)
    } catch (e: any) { setError(e.message); setLoading(false) }
  }

  const stepLabels = ['Agent Setup', 'Benchmark', 'Configure']

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-7">
        <h1 className="text-lg font-semibold text-white mb-0.5">New Benchmark Run</h1>
        <p className="text-sm" style={{ color: '#555' }}>Configure and launch an evaluation against your agent</p>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-1 mb-7">
        {([1, 2, 3] as Step[]).map((n, i) => (
          <div key={n} className="flex items-center gap-1">
            <button
              onClick={() => n < step && setStep(n)}
              className="flex items-center gap-2 text-xs font-medium transition-colors"
              style={{ color: n === step ? '#fff' : n < step ? '#22c55e' : '#444' }}
            >
              <span
                className="w-5 h-5 rounded flex items-center justify-center text-xs font-semibold flex-shrink-0"
                style={{
                  background: n === step ? '#fff' : n < step ? 'rgba(34,197,94,0.15)' : '#161616',
                  color: n === step ? '#000' : n < step ? '#22c55e' : '#444',
                  border: `1px solid ${n === step ? '#fff' : n < step ? 'rgba(34,197,94,0.3)' : '#222'}`,
                }}
              >
                {n < step ? '✓' : n}
              </span>
              {stepLabels[i]}
            </button>
            {n < 3 && <ChevronRight size={12} style={{ color: '#2a2a2a' }} />}
          </div>
        ))}
      </div>

      {/* Step 1: Agent Setup */}
      {step === 1 && (
        <div className="card p-5 space-y-4">
          {/* Sample agent toggle */}
          <div
            className="flex items-center justify-between p-4 rounded cursor-pointer transition-all"
            style={{
              background: useSampleAgent ? 'rgba(34,197,94,0.06)' : '#141414',
              border: `1px solid ${useSampleAgent ? 'rgba(34,197,94,0.2)' : '#222'}`,
              borderRadius: '5px',
            }}
            onClick={() => { setUseSampleAgent(!useSampleAgent); setSelectedBenchmark(null); }}
          >
            <div className="flex items-center gap-3">
              <Bot size={15} style={{ color: useSampleAgent ? '#22c55e' : '#555' }} />
              <div>
                <div className="text-sm font-medium text-white">Use Built-in Demo Agent</div>
                <div className="text-xs mt-0.5" style={{ color: '#555' }}>
                  Auto-starts the sample RAG app on localhost:8001
                </div>
              </div>
            </div>
            <div className={`toggle ${useSampleAgent ? 'on' : ''}`} />
          </div>

          {!useSampleAgent && (
            <>
              <div>
                <label className="block text-xs font-medium mb-1.5" style={{ color: '#555' }}>AGENT URL</label>
                <input
                  value={targetUrl}
                  onChange={e => setTargetUrl(e.target.value)}
                  className="mono"
                  placeholder="http://localhost:8000/solve"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: '#555' }}>REQUEST KEY</label>
                  <input value={requestKey} onChange={e => setRequestKey(e.target.value)}
                    className="mono" placeholder="question" />
                  <p className="text-xs mt-1" style={{ color: '#444' }}>JSON key sent with the question</p>
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: '#555' }}>RESPONSE PATH</label>
                  <input value={responsePath} onChange={e => setResponsePath(e.target.value)}
                    className="mono" placeholder="data" />
                  <p className="text-xs mt-1" style={{ color: '#444' }}>e.g. choices.0.message.content</p>
                </div>
              </div>
            </>
          )}

          <button onClick={() => setStep(2)} className="btn btn-primary w-full justify-center py-2">
            Continue →
          </button>
        </div>
      )}

      {/* Step 2: Benchmark Picker */}
      {step === 2 && (
        <div>
          <div className="card p-5">
            <p className="text-xs mb-4" style={{ color: '#555' }}>
              Select a standard benchmark dataset:
            </p>
            <div className="flex gap-2 overflow-x-auto pb-2 mb-4 scrollbar-hide">
              <button
                onClick={() => setSelectedCategory('All')}
                className={`px-3 py-1 text-xs rounded-full whitespace-nowrap transition-colors ${selectedCategory === 'All' ? 'bg-[#22c55e] text-black font-semibold' : 'bg-[#161616] text-[#888] hover:bg-[#222]'}`}
              >
                All
              </button>
              {categories.map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1 text-xs rounded-full whitespace-nowrap transition-colors ${selectedCategory === cat ? 'bg-[#22c55e] text-black font-semibold' : 'bg-[#161616] text-[#888] hover:bg-[#222]'}`}
                >
                  {cat.toUpperCase()}
                </button>
              ))}
            </div>

            {categories.map(cat => {
              if (selectedCategory !== 'All' && selectedCategory !== cat) return null

              // Filter out benchmarks that don't make sense for the demo agent
              const allowedForDemo = ['gsm8k', 'mmlu', 'truthfulqa', 'toxigen', 'hella-swag', 'winogrande', 'arc']
              const catBenchmarks = benchmarks.filter(b => b.category === cat && (!useSampleAgent || allowedForDemo.includes(b.id)))
              
              if (catBenchmarks.length === 0) return null

              return (
                <div key={cat} className="mb-4">
                  <div className="text-xs font-medium mb-2" style={{ color: '#444' }}>{cat.toUpperCase()}</div>
                  <div className="grid grid-cols-2 gap-1.5">
                    {catBenchmarks.map(b => {
                    const sel = selectedBenchmark === b.id
                    return (
                      <button
                        key={b.id}
                        onClick={() => setSelectedBenchmark(b.id)}
                        className="text-left p-3 rounded transition-all"
                        style={{
                          background: sel ? '#161616' : '#0e0e0e',
                          border: `1px solid ${sel ? '#333' : '#1e1e1e'}`,
                          borderRadius: '5px',
                        }}
                      >
                        <div className="mono text-xs font-semibold text-white mb-1">{b.id}</div>
                        <div className="text-xs leading-snug" style={{ color: '#555' }}>
                          {b.description.split(':')[0]}
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            )})}
          </div>
          <div className="flex gap-2 mt-3">
            <button onClick={() => setStep(1)} className="btn">← Back</button>
            <button
              onClick={() => setStep(3)}
              disabled={!selectedBenchmark}
              className="btn btn-primary flex-1 justify-center"
            >
              Continue →
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Configure */}
      {step === 3 && (
        <div className="space-y-3">
          <div className="card p-5 space-y-5">
            {/* Judge model */}
            <div>
              <label className="block text-xs font-medium mb-2" style={{ color: '#555' }}>JUDGE MODEL</label>
              <div className="grid grid-cols-2 gap-1.5">
                {EVAL_MODELS.map(m => (
                  <button
                    key={m.id}
                    onClick={() => setEvalModel(m.id)}
                    className="text-left px-3 py-2.5 rounded transition-all"
                    style={{
                      background: evalModel === m.id ? '#161616' : '#0e0e0e',
                      border: `1px solid ${evalModel === m.id ? '#333' : '#1e1e1e'}`,
                      borderRadius: '5px',
                    }}
                  >
                    <div className="text-xs font-medium text-white">{m.label}</div>
                    <div className="text-xs" style={{ color: '#555' }}>{m.provider}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Sample count */}
            <div>
              <label className="block text-xs font-medium mb-2" style={{ color: '#555' }}>
                SAMPLES <span className="text-white">{limit}</span>
              </label>
              <input
                type="range" min={1} max={50} value={limit}
                onChange={e => setLimit(+e.target.value)}
                className="w-full"
                style={{ accentColor: '#22c55e' }}
              />
              <div className="flex justify-between text-xs mt-1" style={{ color: '#3a3a3a' }}>
                <span>1 (fast)</span><span>50 (thorough)</span>
              </div>
            </div>

            {/* Chaos toggle */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Zap size={13} style={{ color: enableChaos ? '#ef4444' : '#555' }} strokeWidth={2} />
                  <span className="text-xs font-medium" style={{ color: enableChaos ? '#ef4444' : '#555' }}>
                    CHAOS INJECTION
                  </span>
                </div>
                <div
                  className={`toggle ${enableChaos ? 'on' : ''}`}
                  style={enableChaos ? { background: '#ef4444', borderColor: '#ef4444' } : {}}
                  onClick={() => setEnableChaos(!enableChaos)}
                />
              </div>
              {enableChaos && (
                <div>
                  {isCodingBenchmark && (
                    <div className="flex gap-2 mb-3">
                      <button
                        onClick={() => { setShowCodingChaosOnly(false); setChaosProfile('client_prompt_injection') }}
                        className={`px-2.5 py-1 text-xs rounded-full transition-colors ${
                          !showCodingChaosOnly ? 'bg-[#ef4444] text-white font-semibold' : 'bg-[#161616] text-[#888] hover:bg-[#222]'
                        }`}
                      >
                        All Profiles
                      </button>
                      <button
                        onClick={() => { setShowCodingChaosOnly(true); setChaosProfile('code_context_strip') }}
                        className={`px-2.5 py-1 text-xs rounded-full transition-colors ${
                          showCodingChaosOnly ? 'bg-[#ef4444] text-white font-semibold' : 'bg-[#161616] text-[#888] hover:bg-[#222]'
                        }`}
                      >
                        🖥 Coding
                      </button>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-1.5">
                    {(isCodingBenchmark ? CHAOS_PROFILES : CHAOS_PROFILES.filter(p => p.category === 'general'))
                      .filter(p => !showCodingChaosOnly || p.category === 'coding')
                      .map(p => (
                      <button
                        key={p.id}
                        onClick={() => setChaosProfile(p.id)}
                        className="text-left px-3 py-2 rounded transition-all"
                        style={{
                          background: chaosProfile === p.id ? 'rgba(239,68,68,0.08)' : '#0e0e0e',
                          border: `1px solid ${chaosProfile === p.id ? 'rgba(239,68,68,0.25)' : '#1e1e1e'}`,
                          borderRadius: '5px',
                        }}
                      >
                        {p.category === 'coding' && (
                          <span className="text-xs mr-1" style={{ color: '#f97316' }}>🖥</span>
                        )}
                        <span className="text-xs font-medium text-white">{p.label}</span>
                        <div className="text-xs leading-snug mt-0.5" style={{ color: '#555' }}>{p.description}</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Summary */}
          <div className="card p-4 text-xs space-y-2" style={{ background: '#0d0d0d' }}>
            <div className="flex justify-between items-center">
              <span style={{ color: '#555' }}>Benchmark</span>
              <span className="mono font-medium text-white">{selectedBenchmark}</span>
            </div>
            <div className="flex justify-between items-center">
              <span style={{ color: '#555' }}>Target</span>
              <span className="mono truncate ml-4" style={{ color: '#666' }}>{useSampleAgent ? 'sample rag_app' : targetUrl}</span>
            </div>
            <div className="flex justify-between items-center">
              <span style={{ color: '#555' }}>Judge</span>
              <span style={{ color: '#22c55e' }}>{EVAL_MODELS.find(m => m.id === evalModel)?.label}</span>
            </div>
            <div className="flex justify-between items-center">
              <span style={{ color: '#555' }}>Samples</span>
              <span className="text-white">{limit}</span>
            </div>
            {enableChaos && (
              <div className="flex justify-between items-center">
                <span style={{ color: '#555' }}>Chaos</span>
                <span style={{ color: '#ef4444' }}>{chaosProfile}</span>
              </div>
            )}
          </div>

          {error && (
            <div className="card px-4 py-3 text-xs" style={{ color: '#ef4444', borderColor: 'rgba(239,68,68,0.2)', background: 'rgba(239,68,68,0.06)' }}>
              {error}
            </div>
          )}

          <div className="flex gap-2">
            <button onClick={() => setStep(2)} className="btn">← Back</button>
            <button
              onClick={handleLaunch}
              disabled={loading}
              className={`btn flex-1 justify-center ${enableChaos ? 'btn-danger' : 'btn-primary'}`}
              style={enableChaos ? { background: 'rgba(239,68,68,0.12)', color: '#ef4444', borderColor: 'rgba(239,68,68,0.3)' } : {}}
            >
              {loading
                ? <><div className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />Launching...</>
                : enableChaos
                ? <><Zap size={13} />Launch Chaos Run</>
                : <><FlaskConical size={13} />Launch Benchmark</>
              }
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
