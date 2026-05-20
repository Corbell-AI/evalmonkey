// Static map of benchmark ID → category (mirrors backend SUPPORTED_BENCHMARKS)
export const SUPPORTED_BENCHMARK_CATEGORIES: Record<string, string> = {
  'gsm8k':             'Reasoning',
  'xlam':              'Tool Use',
  'swe-bench':         'Coding',
  'gaia-benchmark':    'Research',
  'human-eval':        'Coding',
  'mmlu':              'Q&A',
  'arc':               'Reasoning',
  'truthfulqa':        'Safety',
  'hella-swag':        'Reasoning',
  'bbh':               'Reasoning',
  'winogrande':        'Q&A',
  'drop':              'Research',
  'natural-questions': 'Q&A',
  'hotpotqa':          'Research',
  'mbpp':              'Coding',
  'apps':              'Coding',
  'mt-bench':          'Instruction Following',
  'alpacaeval':        'Instruction Following',
  'toxigen':           'Safety',
}

export const CHAOS_PROFILES = [
  { id: 'client_prompt_injection',          label: 'Prompt Injection',         description: 'Appends adversarial jailbreak instructions', category: 'general' },
  { id: 'client_typo_injection',            label: 'Typo Injection',            description: 'Obfuscates text with character substitutions', category: 'general' },
  { id: 'client_schema_mutation',           label: 'Schema Mutation',           description: 'Renames JSON request keys to break API parsing', category: 'general' },
  { id: 'client_language_shift',            label: 'Language Shift',            description: 'Appends conflicting language instructions', category: 'general' },
  { id: 'client_payload_bloat',             label: 'Payload Bloat',             description: 'Floods prompt with 10K+ characters to hit token limits', category: 'general' },
  { id: 'client_empty_payload',             label: 'Empty Payload',             description: 'Sends blank string to test graceful rejection', category: 'general' },
  { id: 'client_context_truncation',        label: 'Context Truncation',        description: 'Slices the prompt in half to simulate streaming failure', category: 'general' },
  { id: 'client_unicode_flood',             label: 'Unicode Flood',             description: 'Injects invisible zero-width chars to confuse tokenizers', category: 'general' },
  { id: 'client_role_impersonation',        label: 'Role Impersonation',        description: 'Injects fake SYSTEM OVERRIDE admin escalation', category: 'general' },
  { id: 'client_repetition_loop',           label: 'Repetition Loop',           description: 'Repeats payload 50x to simulate stuck retry loop', category: 'general' },
  { id: 'client_negative_sentiment',        label: 'Hostile Framing',           description: 'Wraps request in angry customer framing', category: 'general' },
  { id: 'client_length_constraint_violation', label: 'Length Constraint',       description: 'Appends conflicting "exactly 2 words" constraint', category: 'general' },
  // Coding-agent-specific
  { id: 'code_context_strip',              label: 'Context Strip',             description: 'Removes code blocks and function signatures from prompt', category: 'coding' },
  { id: 'code_wrong_language',             label: 'Wrong Language',            description: 'Forces response in wrong programming language (JS instead of Python)', category: 'coding' },
  { id: 'code_syntax_break',              label: 'Syntax Break',              description: 'Injects subtle keyword typos to corrupt starter code', category: 'coding' },
  { id: 'code_test_poison',               label: 'Test Poisoning',            description: 'Appends contradictory/impossible test assertions', category: 'coding' },
  { id: 'code_incomplete_signature',      label: 'Incomplete Signature',      description: 'Truncates specification mid-sentence to test ambiguity handling', category: 'coding' },
  { id: 'code_conflicting_constraints',   label: 'Conflicting Constraints',   description: 'Sends logically impossible implementation requirements', category: 'coding' },
]

// Coding-agent-relevant chaos profiles for quick selection
export const CODING_CHAOS_PROFILES = CHAOS_PROFILES.filter(p => p.category === 'coding')

export const EVAL_MODELS = [
  // AWS Bedrock (long-term key via BEDROCK_API_KEY)
  { id: 'bedrock/anthropic.claude-3-haiku-20240307-v1:0',  label: 'Claude Haiku 3',    provider: 'AWS Bedrock' },
  { id: 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0', label: 'Claude Sonnet 3.5', provider: 'AWS Bedrock' },
  { id: 'bedrock/anthropic.claude-3-5-haiku-20241022-v1:0',  label: 'Claude Haiku 3.5', provider: 'AWS Bedrock' },
  // OpenAI
  { id: 'gpt-4o',                         label: 'GPT-4o',             provider: 'OpenAI' },
  { id: 'gpt-4o-mini',                    label: 'GPT-4o Mini',        provider: 'OpenAI' },
  // Anthropic direct
  { id: 'anthropic/claude-haiku-4-5',     label: 'Claude Haiku 4.5',   provider: 'Anthropic' },
  { id: 'anthropic/claude-sonnet-4-5',    label: 'Claude Sonnet 4.5',  provider: 'Anthropic' },
  // Local
  { id: 'ollama/llama3',                  label: 'Llama 3 (Ollama)',   provider: 'Ollama' },
]
