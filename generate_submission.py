#!/usr/bin/env python3
"""
Generate submission.jsonl by composing messages for all 25 seed triggers.
Uses the bot's composer engine directly (no HTTP needed).

Usage: python generate_submission.py
Output: submission.jsonl
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from bot import VeraComposer

DATASET_DIR = Path(__file__).parent / "dataset"


def load_dataset():
    """Load all seed data."""
    categories = {}
    cat_dir = DATASET_DIR / "categories"
    for f in cat_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            categories[data["slug"]] = data

    with open(DATASET_DIR / "merchants_seed.json") as fp:
        merchants_data = json.load(fp)
    merchants = {m["merchant_id"]: m for m in merchants_data["merchants"]}

    with open(DATASET_DIR / "customers_seed.json") as fp:
        customers_data = json.load(fp)
    customers = {c["customer_id"]: c for c in customers_data["customers"]}

    with open(DATASET_DIR / "triggers_seed.json") as fp:
        triggers_data = json.load(fp)
    triggers = {t["id"]: t for t in triggers_data["triggers"]}

    return categories, merchants, customers, triggers


def main():
    import time
    categories, merchants, customers, triggers = load_dataset()
    composer = VeraComposer()

    results = []
    for i, (trigger_id, trigger) in enumerate(triggers.items(), 1):
        merchant_id = trigger.get("merchant_id", "")
        customer_id = trigger.get("customer_id")

        merchant = merchants.get(merchant_id, {})
        category_slug = merchant.get("category_slug", "")
        category = categories.get(category_slug, {})
        customer = customers.get(customer_id) if customer_id else None

        # Compose with retry
        result = None
        for attempt in range(3):
            try:
                result = composer.compose(category, merchant, trigger, customer)
                if result and result.get("body"):
                    break
            except Exception as e:
                print(f"    [retry {attempt+1}] {e}")
                time.sleep(2)
        if not result:
            result = composer._fallback(category, merchant, trigger, customer)

        entry = {
            "test_id": f"T{i:02d}",
            "trigger_id": trigger_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "body": result.get("body", ""),
            "cta": result.get("cta", "open_ended"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", ""),
            "rationale": result.get("rationale", "")
        }
        results.append(entry)
        print(f"  [T{i:02d}] {trigger.get('kind', '?'):30s} → {merchant.get('identity', {}).get('name', '?')[:25]}")

    # Write JSONL
    output_path = Path(__file__).parent / "submission.jsonl"
    with open(output_path, "w", encoding="utf-8") as fp:
        for entry in results:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✓ Written {len(results)} entries to {output_path}")


if __name__ == "__main__":
    main()
