from __future__ import annotations

import argparse
import json
import os
import sys

# Suppress ML framework warnings (we use API-based models, not local ones)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .runner import FindBatchParams, FindOneParams, find_batch, find_one, resume


def _pp(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rule-book-agent", description="Agentic rulebook finder (ad-hoc CLI).")
    sub = p.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("find-one", help="Find and download a rulebook for a single game.")
    one.add_argument("--game-name", type=str, default=None)
    one.add_argument("--bgg-id", type=int, default=None)
    one.add_argument("--db-path", type=str, default="")
    one.add_argument("--recursion-limit", type=int, default=50, help="Max agent steps (default: 50)")

    batch = sub.add_parser("find-batch", help="Find and download rulebooks for a rank range (skips existing).")
    batch.add_argument("--rank-from", type=int, default=1)
    batch.add_argument("--rank-to", type=int, default=50)
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--db-path", type=str, default="")
    batch.add_argument("--recursion-limit", type=int, default=50, help="Max agent steps (default: 50)")

    res = sub.add_parser("resume", help="Resume a paused run (HITL).")
    res.add_argument("--run-id", type=str, required=True)
    res.add_argument("--recursion-limit", type=int, default=30)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "find-one":
        out = find_one(
            FindOneParams(
                game_name=args.game_name,
                bgg_id=args.bgg_id,
                db_path=args.db_path,
                recursion_limit=args.recursion_limit,
            )
        )
        print(_pp(out))
        if not out.get("skipped"):
            print("\nLangSmith tracing (optional):")
            print("  export LANGSMITH_API_KEY=your-key")
            print("  export LANGCHAIN_TRACING_V2=true")
            print("  Project: LANGCHAIN_PROJECT=boardgame-rulebook-finder (auto-set if missing)")
        if out.get("run_paused"):
            print("\nRun paused (human-in-the-loop).")
            print(f"run_id: {out.get('run_id')}")
            print(f"reason: {out.get('pause_reason')}")
            print("\nAfter you resolve the blocking step in the opened browser session, run:")
            print(f"  rule-book-agent resume --run-id {out.get('run_id')}\n")
        return 0

    if args.cmd == "find-batch":
        out = find_batch(
            FindBatchParams(
                rank_from=args.rank_from,
                rank_to=args.rank_to,
                limit=args.limit,
                db_path=args.db_path,
                recursion_limit=args.recursion_limit,
            )
        )
        
        # Formatted batch output
        print(f"\n📊 Batch Results: Ranks {args.rank_from}-{args.rank_to}")
        print(f"Total games in range: {out.get('total', 0)}")
        print(f"Missing rulebooks: {out.get('missing', 0)}")
        print(f"Already have rulebooks: {out.get('total', 0) - out.get('missing', 0)}")
        
        results = out.get('results', [])
        if results:
            print(f"\nPer-game results ({len(results)} processed):")
            print("-" * 80)
            
            success_count = 0
            paused_count = 0
            failed_count = 0
            
            for i, result in enumerate(results, 1):
                game_name = result.get('game_name', 'Unknown')
                game_rank = result.get('game_rank')
                run_id = result.get('run_id', 'N/A')
                validated = result.get('validated_rulebook')
                paused = result.get('run_paused', False)
                
                if validated:
                    status = "✅ Success"
                    success_count += 1
                    file_path = validated.get('file_path', 'N/A')
                    print(f"  {i}. [{game_rank}] {game_name}: {status}")
                    print(f"     📄 File: {file_path}")
                    print(f"     🔗 URL: {validated.get('url', 'N/A')}")
                elif paused:
                    status = "⏸️  Paused (HITL)"
                    paused_count += 1
                    print(f"  {i}. [{game_rank}] {game_name}: {status}")
                    print(f"     Reason: {result.get('pause_reason', 'Unknown')}")
                    print(f"     Resume: rule-book-agent resume --run-id {run_id}")
                else:
                    status = "❌ Failed"
                    failed_count += 1
                    print(f"  {i}. [{game_rank}] {game_name}: {status}")
                    print(f"     Run ID: {run_id}")
                    print(f"     Check: {result.get('run_dir', 'N/A')}/final_state.json")
                
                print()
            
            print("-" * 80)
            print(f"Summary: {success_count} ✅ | {paused_count} ⏸️  | {failed_count} ❌")
        else:
            print("\n✅ All games in this range already have rulebooks!")
        
        print("\nLangSmith tracing (optional):")
        print("  export LANGSMITH_API_KEY=your-key")
        print("  export LANGCHAIN_TRACING_V2=true")
        print("  Project: boardgame-rulebook-finder (auto-set)")
        
        return 0

    if args.cmd == "resume":
        out = resume(args.run_id, recursion_limit=args.recursion_limit)
        print(_pp(out))
        return 0

    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


