import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

# Fix tokenizer parallelism warnings from Selenium forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .workflow import RulebookWorkflow, RunConfig, load_or_init_context, save_context
from llama_index.core.workflow import StartEvent
from dotenv import load_dotenv


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging to both console and file"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # More detailed file logging
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
        
    return root_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LlamaIndex rulebook workflow")
    p.add_argument("--db", default="bgg_games.db", help="Path to SQLite DB")
    p.add_argument("--rank-from", type=int, required=True, help="Start rank inclusive")
    p.add_argument("--rank-to", type=int, required=True, help="End rank inclusive")
    p.add_argument("--model-strategy", default="mlx-llm", help="Model strategy key, used to partition Context")
    p.add_argument("--ctx-dir", default="bgg_data/llama_index_workflow/state", help="Context directory")
    p.add_argument("--log-dir", default="bgg_data/llama_index_workflow/logs", help="Logs directory")
    p.add_argument("--rulebooks-dir", default="bgg_data/llama_index_workflow/rulebooks", help="Output rulebooks directory")
    p.add_argument("--summary-json", default="bgg_data/llama_index_workflow/summary.json", help="Path to write JSON summary")
    return p.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    cfg = RunConfig(
        db_path=Path(args.db),
        rulebooks_dir=Path(args.rulebooks_dir),
        model_strategy=args.model_strategy,
        context_dir=Path(args.ctx_dir),
        log_dir=Path(args.log_dir),
    )
    
    # Setup file logging
    log_file = cfg.log_dir / f"workflow_{cfg.model_strategy}.log"
    setup_logging(log_file)

    wf = RulebookWorkflow(cfg)
    ctx = load_or_init_context(wf, cfg.context_dir, cfg.model_strategy)

    start_ev = StartEvent(payload={"rank_from": args.rank_from, "rank_to": args.rank_to})
    if ctx is not None:
        handler = wf.run(ctx=ctx, start_event=start_ev)
    else:
        handler = wf.run(start_event=start_ev)

    # Await completion; logs from steps will show progressively.
    result = await handler

    # Persist context for this model strategy
    save_context(handler.ctx, cfg.context_dir, cfg.model_strategy)

    # Write summary
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Summary written to {args.summary_json}")


def main() -> None:
    # Load .env from project root
    try:
        load_dotenv()
    except Exception:
        pass
    
    args = parse_args()
    
    # Setup basic console logging first
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()


