"""Train a sparse byte n-gram language model on the visible TinyStories split."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


VOCAB_SIZE = 257


def _context_key(context):
    return ",".join(str(token) for token in context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--smoothing", type=float, required=True)
    parser.add_argument("--max-contexts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.order <= 5:
        raise ValueError("order must be between 1 and 5")
    if args.smoothing <= 0:
        raise ValueError("smoothing must be positive")
    if args.max_contexts <= 0:
        raise ValueError("max-contexts must be positive")

    train_path = Path("/benchmark/train/train.json")
    sequences = json.loads(train_path.read_text(encoding="utf-8"))["sequences"]
    tables = [defaultdict(dict) for _ in range(args.order)]
    targets = 0
    for sequence in sequences:
        for position in range(1, len(sequence)):
            target = sequence[position]
            targets += 1
            for width in range(min(args.order - 1, position) + 1):
                context = tuple(sequence[position - width : position]) if width else ()
                counts = tables[width][context]
                counts[target] = counts.get(target, 0) + 1

    serialized_tables = []
    for width, table in enumerate(tables):
        items = list(table.items())
        if width >= 2 and len(items) > args.max_contexts:
            items.sort(key=lambda item: (-sum(item[1].values()), item[0]))
            items = items[: args.max_contexts]
        items.sort(key=lambda item: item[0])
        serialized_tables.append(
            {
                _context_key(context): sorted(counts.items())
                for context, counts in items
            }
        )

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    model = {
        "type": "sparse_byte_ngram",
        "vocab_size": VOCAB_SIZE,
        "order": args.order,
        "smoothing": args.smoothing,
        "training_targets": targets,
        "contexts_per_width": [len(table) for table in serialized_tables],
        "tables": serialized_tables,
    }
    (args.output / "model.json").write_text(
        json.dumps(model, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        Path("/workspace/candidate_template.py"), args.output / "candidate.py"
    )
    print(
        f"trained order={args.order} targets={targets} "
        f"contexts={model['contexts_per_width']}"
    )


if __name__ == "__main__":
    main()
