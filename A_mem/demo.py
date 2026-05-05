from agentic_memory.memory_system import AgenticMemorySystem


# OR initialize with Ollama 🚀
memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend="ollama",
    llm_model="qwq:latest"
)

# Add Memories with Automatic LLM Analysis ✨
# Simple addition - LLM automatically generates keywords, context, and tags
memory_id1 = memory_system.add_note(
    "Machine learning algorithms use neural networks to process complex datasets and identify patterns."
)

# Check the automatically generated metadata
memory = memory_system.read(memory_id1)
print(f"Content: {memory.content}")
print(f"Auto-generated Keywords: {memory.keywords}")  # e.g., ['machine learning', 'neural networks', 'datasets']
print(f"Auto-generated Context: {memory.context}")    # e.g., "Discussion about ML algorithms and data processing"
print(f"Auto-generated Tags: {memory.tags}")          # e.g., ['artificial intelligence', 'data science', 'technology']

# Partial metadata provision - LLM fills in missing attributes
memory_id2 = memory_system.add_note(
    content="Python is excellent for data science applications",
    keywords=["Python", "programming"]  # Provide keywords, LLM will generate context and tags
)

# Manual metadata provision - no LLM analysis needed
memory_id3 = memory_system.add_note(
    content="Project meeting notes for Q1 review",
    keywords=["meeting", "project", "review"],
    context="Business project management discussion",
    tags=["business", "project", "meeting"],
    timestamp="202503021500"  # YYYYMMDDHHmm format
)

# Enhanced Retrieval with Metadata 🔍
# The system now uses generated metadata for better semantic search
results = memory_system.search("artificial intelligence data processing", k=3)
for result in results:
    print(f"ID: {result.get('id', 'N/A')}")
    print(f"Content: {result.get('content', 'N/A')}")
    print(f"Keywords: {result.get('keywords', 'N/A')}")
    print(f"Tags: {result.get('tags', 'N/A')}")
    print(f"Relevance Score: {result.get('score', 'N/A')}")
    print("---")

# Alternative search methods
results = memory_system.search_agentic("neural networks", k=5)
for result in results:
    print(f"ID: {result.get('id', 'N/A')}")
    print(f"Content: {result.get('content', 'N/A')}")
    print(f"Tags: {result.get('tags', 'N/A')}")
    print("---")

# Update Memories 🔄
memory_system.update(memory_id1, content="Updated: Deep learning neural networks for pattern recognition")

# Delete Memories ❌
memory_system.delete(memory_id3)

# Memory Evolution 🧬
# The system automatically evolves memories by:
# 1. Using LLM to analyze content and generate semantic metadata
# 2. Finding relationships using enhanced ChromaDB embeddings (content + metadata)
# 3. Updating tags, context, and connections based on related memories
# 4. Creating semantic links between memories
# This happens automatically when adding or updating memories!
