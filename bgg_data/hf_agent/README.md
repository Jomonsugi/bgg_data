# HF Agent - Agentic Rulebook Fetcher

An experimental agentic approach to fetching board game rulebooks using Hugging Face's SmolAgents framework with web browser automation.

## Overview

The HF Agent approach uses an autonomous AI agent to search for and download board game rulebooks. Unlike the rule-based DAG approach, this method gives the AI agent more autonomy to make decisions about how to navigate websites, interpret content, and solve problems dynamically.

## Key Features

- **Autonomous Decision Making**: Agent decides how to navigate and interact with websites
- **Browser Automation**: Uses Selenium with Helium for web interactions
- **Screenshot Analysis**: Takes screenshots at each step for visual context
- **Dynamic Problem Solving**: Adapts strategies based on what it encounters
- **PDF Download Integration**: Includes robust PDF download capabilities

## Architecture

```
hf_agent/
├── web_search_agent.py     # Main agent script with tools
├── hf_rulebooks/          # Downloaded rulebooks
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Chrome browser installed
- Hugging Face API access or local models
- Environment variables (optional):
  - `HUGGINGFACE_API_KEY` for HF models
  - `OPENAI_API_KEY` for OpenAI models

### Basic Usage

```bash
# Search for a specific board game rulebook
uv run python web_search_agent.py "Brass Birmingham"

# With custom model
uv run python web_search_agent.py "Pandemic Legacy" --model-type "OpenAIServerModel" --model-id "gpt-4"

# With custom prompt
uv run python web_search_agent.py "Catan" --prompt "Find and download the official rulebook for {board_game}"
```

## How It Works

### 1. **Agent Initialization**
- Loads specified language model (Together.ai, OpenAI, or local)
- Initializes Chrome browser with automation capabilities
- Sets up screenshot capture for visual feedback

### 2. **Autonomous Web Search**
- Agent performs web searches using DuckDuckGo
- Navigates to promising websites autonomously
- Makes decisions about which links to follow

### 3. **Dynamic Website Navigation**
- Agent analyzes page screenshots to understand content
- Clicks buttons, follows links, and navigates forms
- Adapts to different website layouts and structures

### 4. **Intelligent Content Recognition**
- Uses visual analysis to identify download buttons
- Recognizes PDF links and rulebook-related content
- Handles pop-ups, cookie banners, and other obstacles

### 5. **Robust PDF Download**
- Multiple download strategies with fallbacks
- Handles redirects, authentication, and special cases
- Validates downloaded files are actual PDFs
- Converts special URLs (Dropbox, Google Drive) to direct downloads

## Agent Tools

The agent has access to several specialized tools:

### Web Interaction Tools
- `DuckDuckGoSearchTool()`: Perform web searches
- `go_back()`: Navigate back in browser
- `close_popups()`: Handle modal dialogs and pop-ups
- `search_item_ctrl_f()`: Search for text on current page

### Download Tools
- `download_pdf()`: Download PDFs with robust error handling
- `get_current_url()`: Get current browser URL for context

### Browser Automation (Helium)
The agent can use all Helium commands:
```python
go_to('website.com')           # Navigate to URL
click('Download PDF')          # Click elements by text
click(Link('Rules'))          # Click links specifically
scroll_down(num_pixels=1200)  # Scroll page
Text('Accept cookies').exists() # Check if text exists
```

## Configuration

### Model Options

**Together.ai (Default)**
```bash
export TOGETHER_API_KEY="your_key"
python web_search_agent.py "Game Name"
```

**OpenAI**
```bash
export OPENAI_API_KEY="your_key"  
python web_search_agent.py "Game Name" --model-type "OpenAIServerModel" --model-id "gpt-4o"
```

**Local Models**
```bash
python web_search_agent.py "Game Name" --model-type "TransformersModel" --model-id "microsoft/DialoGPT-medium"
```

### Browser Settings
The agent runs Chrome with these settings:
- Window size: 1000x1350
- PDF viewer disabled (for direct downloads)
- Non-headless mode (visible browser for debugging)
- Device scale factor normalized

## Prompt Engineering

The default prompt guides the agent to:
1. Search for the board game's official website first
2. Look for rulebook downloads on official sites
3. Avoid starting with "rulebook" searches initially
4. Verify downloads by checking the actual content

### Custom Prompts
```bash
python web_search_agent.py "Wingspan" --prompt "
Search for {board_game} and find the official PDF rulebook.
Focus on the publisher's website first.
Download any PDFs you find that appear to be rulebooks.
"
```

## Outputs

### Downloaded Files
- **Location**: `hf_rulebooks/`
- **Naming**: Uses provided filename or auto-generates
- **Format**: Validated PDF files only

### Screenshots
- Automatically captures browser state after each action
- Stored in agent memory for decision making
- Helps with debugging and understanding agent behavior

### Logs
- Detailed step-by-step execution logs
- Shows agent reasoning and decision process
- Includes error messages and retry attempts

## Advanced Features

### PDF Download Capabilities
- **Multiple Retry Strategies**: Exponential backoff on failures
- **Special URL Handling**: Converts Dropbox/Google Drive sharing links
- **Browser Context Integration**: Uses cookies and referer headers
- **Content Validation**: Ensures downloads are actual PDFs
- **HTML Link Extraction**: Finds PDF links embedded in HTML pages

### Error Recovery
- **Pop-up Handling**: Automatically closes modal dialogs
- **Navigation Recovery**: Can go back and try different approaches
- **Search Fallbacks**: Multiple search strategies if initial attempts fail
- **Download Fallbacks**: Multiple download methods with different headers

## Comparison with DAG Approach

| Aspect | HF Agent | DAG |
|--------|----------|-----|
| **Decision Making** | Autonomous, adaptive | Rule-based, predictable |
| **Website Handling** | Dynamic adaptation | Structured handlers |
| **Debugging** | Visual screenshots | Structured logs |
| **Reliability** | Variable, creative | Consistent, systematic |
| **Customization** | Prompt engineering | Code modification |
| **Speed** | Variable (agent thinking) | Faster (direct execution) |

## Troubleshooting

### Common Issues

1. **Agent Gets Stuck**: The agent may get confused on complex sites
   - Solution: Use `--screenshots` to see what the agent sees
   - Try more specific prompts

2. **Model API Errors**: Authentication or rate limiting issues
   - Check API keys are set correctly
   - Try local models as fallback

3. **Browser Issues**: Chrome automation problems
   - Ensure Chrome is installed and up to date
   - Check ChromeDriver compatibility

4. **Download Failures**: PDFs not downloading
   - Agent will try multiple strategies automatically
   - Check network connectivity and site accessibility

### Debugging Tips

```bash
# Run with visible browser and detailed logging
python web_search_agent.py "Game Name" --model-type "TransformersModel" --model-id "microsoft/DialoGPT-medium"

# Use simpler local model for testing
python web_search_agent.py "Simple Game" --model-type "TransformersModel" --model-id "microsoft/DialoGPT-medium"
```

## Future Enhancements

- **Multi-agent Collaboration**: Multiple agents working together
- **Learning from Experience**: Agent memory of successful strategies
- **Site-specific Strategies**: Specialized approaches for known sites
- **Quality Assessment**: Agent evaluation of downloaded rulebooks
- **Batch Processing**: Process multiple games in sequence

## Contributing

This is an experimental framework for testing agentic approaches to rulebook fetching. The autonomous nature makes it:

- **Flexible**: Adapts to new website layouts automatically
- **Creative**: Can discover new sources and strategies
- **Unpredictable**: May find solutions humans didn't consider
- **Challenging to Debug**: Requires understanding agent decision-making

The agentic approach complements the rule-based DAG system by providing a more adaptive but less predictable alternative for difficult cases.
