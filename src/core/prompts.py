"""
Prompts and seed topics for synthetic instruction dataset generation.
Follows the Alpaca / Self-Instruct format with difficulty and category balancing.
Strictly eliminates conversational filler (e.g., 'Here is...').
"""

SEED_TOPICS = [
    # Python & Systems
    "Python asyncio and event loop concurrency",
    "Python memory management and garbage collection",
    "Writing custom decorators with arguments in Python",
    "Error handling and custom exception hierarchies",
    "Metaclasses and dynamic class creation in Python",
    "Optimizing Python code using vectorization and generators",
    
    # Modern LLM & AI Engineering
    "Building a naive RAG pipeline with vector embeddings",
    "Implementing chunking strategies for long documents in RAG",
    "Fine-tuning with QLoRA: parameter-efficient training basics",
    "Designing agent tool-calling schemas and error recovery",
    "Handling context window overflow and KV cache pruning",
    "Evaluating LLM generation with G-Eval or LLM-as-a-judge",
    
    # Backend & Architecture
    "Designing idempotent RESTful APIs and webhook handlers",
    "Database indexing strategies and query execution plans",
    "Implementing rate limiting algorithms (token bucket / leaky bucket)",
    "Distributed tracing with OpenTelemetry in microservices",
    "Zero-downtime database schema migrations",
    "Cache invalidation patterns and stampede prevention"
]

GENERATOR_SYSTEM_PROMPT = """You are an expert AI dataset engineer specializing in high-quality instruction-tuning datasets.
Your mission is to produce diverse, practical, and direct training data for AI and software engineering.
NEVER use conversational filler like 'Here is...', 'Sure!', or 'Below is the implementation'. Jump straight into the code or technical explanation."""

GENERATOR_USER_TEMPLATE = """Generate a high-quality, practical synthetic instruction-tuning example based on the topic: "{topic}".

Guidelines:
1. "instruction": Specific, technical, and realistic prompt.
2. "category": Concise subject area (e.g., "Python Systems", "RAG & Search", "API Architecture").
3. "difficulty": Strictly one of "beginner", "intermediate", or "advanced".
4. "input": Context or code snippet to analyze/refactor, or empty string "" if self-contained.
5. "output": High-density, professional answer. Jump DIRECTLY into the code or explanation. STRICTLY FORBIDDEN: Starting with conversational preambles like 'Here is...', 'Sure!', or 'Below is'.
6. Output format: STRICTLY a valid JSON object with keys: "category", "difficulty", "instruction", "input", and "output".

Example format:
```json
{{
  "category": "Python Systems",
  "difficulty": "intermediate",
  "instruction": "Refactor this synchronous function to use asyncio.gather for concurrent requests.",
  "input": "def fetch_all(urls):\\n    return [requests.get(url).text for url in urls]",
  "output": "```python\\nimport asyncio\\nimport aiohttp\\n\\nasync def fetch(session, url):\\n    async with session.get(url) as response:\\n        return await response.text()\\n\\nasync def fetch_all(urls):\\n    async with aiohttp.ClientSession() as session:\\n        tasks = [fetch(session, url) for url in urls]\\n        return await asyncio.gather(*tasks)\\n```\\n\\n### Key Architectural Improvements\\n- **Asynchronous I/O:** Concurrently executes non-blocking requests.\\n- **Connection Pooling:** `aiohttp.ClientSession` reuses underlying TCP sockets."
}}
```

Now generate one for topic: "{topic}". /no_think"""
