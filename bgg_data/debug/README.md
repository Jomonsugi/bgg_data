# Debug Directory

This directory contains debug information for troubleshooting LLM assessment failures in the rulebook fetcher.

## Directory Structure

```
debug/
├── README.md                    # This file
└── llm_assessments/            # LLM assessment debug information
    ├── YYYYMMDD_HHMMSS_GameName_filename_assessment.json    # LLM assessment details
    ├── YYYYMMDD_HHMMSS_GameName_filename_assessment.png     # Screenshot when assessment failed
    └── YYYYMMDD_HHMMSS_GameName_filename_assessment.json    # Screenshot metadata
```

## Debug Information

### LLM Assessment JSON Files

When the LLM assessment fails (file is not marked as official or English), a JSON file is saved containing:

- **timestamp**: When the assessment was performed
- **game_name**: Name of the game being processed
- **file_path**: Path to the file that was assessed
- **file_size_mb**: Size of the file in MB
- **file_type**: File extension (.pdf, .html, etc.)
- **llm_prompt**: The exact prompt sent to the LLM
- **llm_response**: The raw response from the LLM
- **assessment_result**: The parsed assessment (is_official, is_english)
- **sample_text_preview**: Preview of the text extracted from the file
- **sample_text_length**: Length of the extracted text

### Debug Screenshots

When LLM assessment fails and screenshots are enabled, a PNG file is saved showing:

- The current webpage being processed
- What the LLM was looking at when it made its decision
- The context around the failed assessment

### Screenshot Metadata

Each screenshot has a corresponding JSON file with:

- **timestamp**: When the screenshot was taken
- **game_name**: Name of the game
- **file_path**: Path to the file that failed assessment
- **source_url**: URL where the file was downloaded from
- **llm_assessment**: The assessment results that caused the failure
- **screenshot_path**: Path to the screenshot file
- **current_page_url**: URL of the page when screenshot was taken

## Usage

1. **Enable screenshots** in your rulebook fetcher configuration
2. **Run the fetcher** - debug information will be automatically saved when assessments fail
3. **Review the debug files** to understand why files are being rejected
4. **Use the information** to improve the LLM prompts or fix issues in the assessment logic

## Example Debug Session

```
# Run rulebook fetcher with screenshots enabled
python -m bgg_data.cli.main fetch-rulebooks --rank-from 1 --rank-to 10 --screenshots

# Check debug directory for failed assessments
ls -la bgg_data/debug/llm_assessments/

# Review a specific assessment failure
cat bgg_data/debug/llm_assessments/20241201_143022_Gloomhaven_rules_assessment.json

# View the screenshot that caused the failure
open bgg_data/debug/llm_assessments/20241201_143022_Gloomhaven_rules_assessment.png
```

## Troubleshooting

- **No debug files created**: Ensure screenshots are enabled and LLM assessment is failing
- **Screenshot capture fails**: Check if the web handler is available and working
- **JSON parsing errors**: The LLM response may be malformed - check the raw response
- **File size issues**: Very large files may cause memory issues during text extraction
