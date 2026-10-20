"""
Prompts and seed topics for synthetic instruction dataset generation.
Follows the Alpaca / Self-Instruct format with difficulty and category balancing.
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
Your mission is to produce diverse, practical, and syntactically flawless training data for AI and software engineering."""

GENERATOR_USER_TEMPLATE = """Generate a high-quality, practical synthetic instruction-tuning example based on the topic: "{topic}".

Guidelines:
1. Make the instruction specific, technical, and realistic (avoid generic tutorials).
2. "category": Concise subject area (e.g., "Python Systems", "RAG & Search", "API Architecture").
3. "difficulty": Strictly one of "beginner", "intermediate", or "advanced".
4. "input": Context or code snippet to analyze/refactor, or empty string "" if self-contained.
5. "output": A clear, complete answer with syntactically valid and runnable code.
6. Output STRICTLY a valid JSON object with keys: "category", "difficulty", "instruction", "input", and "output".

Example format:
```json
{{
  "category": "Python Systems",
  "difficulty": "intermediate",
  "instruction": "Refactor this synchronous function to use asyncio.gather for concurrent requests.",
  "input": "def fetch_all(urls):\\n    return [requests.get(url).text for url in urls]",
  "output": "Here is the refactored asynchronous version using `aiohttp` and `asyncio.gather`:\\n\\n```python\\nimport asyncio\\nimport aiohttp\\n...\\n```"
}}
```

Now generate one for topic: "{topic}". /no_think"""
