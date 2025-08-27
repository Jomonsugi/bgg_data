# BGG Data - Rulebook Fetching Experiments

A collection of experimental approaches for automatically finding and downloading board game rulebooks using different AI and automation strategies.

## Overview

This project explores various frameworks and techniques for automatically collecting board game rulebooks from the web. Each directory represents a different experimental approach, allowing for comparison and evaluation of different strategies.

## Project Structure

### 🗄️ [`database/`](bgg_data/database/README.md)
**Shared Database Layer**
- BoardGameGeek (BGG) game data collection and storage
- SQLite database shared across all experimental approaches
- Game metadata, rankings, and publisher information
- Common data layer for consistent game information

### 📊 [`dag/`](bgg_data/dag/README.md) 
**Rule-Based Approach**
- Structured, deterministic workflow using LLM vision models
- Rule-based decision making with predictable fallbacks
- Selenium web automation with screenshot analysis
- Multiple vision backends (Together.ai, local MLX)
- Robust error handling and retry mechanisms

### 🤖 [`hf_agent/`](bgg_data/hf_agent/README.md)
**Agentic Approach**
- Autonomous AI agent using Hugging Face SmolAgents
- Dynamic decision making and adaptive problem solving  
- Browser automation with visual context understanding
- Agent-driven web navigation and content discovery
- Experimental autonomous behavior patterns

## Quick Start

### Prerequisites
- Python 3.11+
- uv package manager
- Chrome browser + ChromeDriver

### Installation
```bash
# Clone and install dependencies
git clone <repository>
cd bgg_data
uv sync
```

### Collect Game Data (Required First Step)
```bash
# Collect BGG data for top 100 games
uv run python -m bgg_data.database.collect_data --limit 100
```

### Choose Your Approach

**Rule-Based (DAG)**
```bash
# Process games with structured workflow
uv run python -m dag.cli.main --rank-from 1 --rank-to 20
```

**Agentic (HF Agent)**  
```bash
# Let AI agent search autonomously
uv run python hf_agent/web_search_agent.py "Brass Birmingham"
```

## Experimental Approaches

### Rule-Based (DAG)
**Best for**: Predictable results, debugging, production use
- ✅ Consistent behavior
- ✅ Clear error handling  
- ✅ Efficient processing
- ❌ Less adaptable to new sites

### Agentic (HF Agent)
**Best for**: Exploration, handling edge cases, research
- ✅ Adaptive to new layouts
- ✅ Creative problem solving
- ✅ Autonomous discovery
- ❌ Unpredictable behavior
- ❌ Harder to debug

## Shared Components

### Database Layer
All approaches use the same SQLite database:
- Game metadata from BGG API
- Ranking and publisher information
- Shared game identifiers
- Consistent data access patterns

### Output Standards
- **Rulebooks**: Standardized PDF filenames
- **Logs**: Structured logging across approaches
- **Screenshots**: Debug visual information
- **Metadata**: Download success tracking

## API Keys & Configuration

### Optional but Recommended
```bash
# For LLM vision models
export TOGETHER_API_KEY="your_key"
export HUGGINGFACE_API_KEY="your_key" 
export OPENAI_API_KEY="your_key"

# For web search fallbacks
export TAVILY_API_KEY="your_key"
```

### Local Alternatives
```bash
# Use local MLX models (no API keys needed)
VISION_BACKEND=mlx uv run python -m dag.cli.main --rank-from 1 --rank-to 5
```

## Performance Comparison

| Metric | DAG (Rule-Based) | HF Agent (Agentic) |
|--------|------------------|---------------------|
| **Speed** | Fast (structured) | Variable (thinking time) |
| **Success Rate** | High (predictable) | Variable (adaptive) |
| **Debugging** | Easy (logs) | Moderate (screenshots) |
| **Maintenance** | Code changes | Prompt engineering |
| **Scalability** | Excellent | Good |

## Future Experiments

The modular structure makes it easy to add new approaches:

### Planned Additions
- **`selenium_pure/`** - Pure Selenium without LLM vision
- **`playwright_agent/`** - Playwright-based automation
- **`api_first/`** - API-focused approach with web fallback
- **`ml_classifier/`** - Machine learning classification of download links
- **`hybrid/`** - Combination of multiple approaches

### Research Areas
- Multi-agent collaboration
- Reinforcement learning from successful downloads
- Site-specific strategy learning
- Quality assessment of downloaded rulebooks
- Cross-validation between approaches

## Contributing

Each experimental approach is self-contained but shares common interfaces:

1. **Game Input**: Uses shared `Game` objects from database
2. **Output Format**: Standardized PDF naming and location
3. **Logging**: Consistent log formats for comparison
4. **Configuration**: Shared environment variables where possible

### Adding New Approaches
1. Create new directory (e.g., `my_approach/`)
2. Add README.md explaining the approach
3. Implement game processing with shared database
4. Follow output standards for comparison
5. Update this root README with the new approach

## Evaluation & Metrics

### Success Metrics
- **Download Success Rate**: Percentage of games with successful rulebook downloads
- **Processing Speed**: Games processed per minute
- **Error Recovery**: Ability to handle failed attempts
- **Content Quality**: Validation of downloaded PDFs

### Comparison Tools
- Cross-approach validation of results
- Performance benchmarking scripts
- Success rate analysis
- Error pattern identification

## License

This project is experimental research code. See [LICENSE](LICENSE) for details.