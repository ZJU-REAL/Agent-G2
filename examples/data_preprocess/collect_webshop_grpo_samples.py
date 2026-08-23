import argparse
import itertools
import json
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

from gmsv.webshop import goal_to_target_key, target_key_to_string


PRICE_RANGE = [10.0 * i for i in range(1, 100)]


def _load_short_products(data_dir: Path, catalog_random_seed: int):
    random.seed(catalog_random_seed)
    with open(data_dir / "items_shuffle_1000.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    with open(data_dir / "items_ins_v2_1000.json", "r", encoding="utf-8") as f:
        attributes = json.load(f)

    asins = set()
    all_products = []
    for idx, product in enumerate(products):
        asin = product["asin"]
        if asin == "nan" or len(asin) > 10 or asin in asins:
            continue
        asins.add(asin)

        item = dict(product)
        item["category"] = product["category"]
        item["query"] = product["query"]
        item["product_category"] = product["product_category"]
        item["Title"] = product["name"]
        item["Description"] = product["full_description"]
        item["Reviews"] = []
        item["Rating"] = "N.A."
        item["BulletPoints"] = (
            product["small_description"]
            if isinstance(product["small_description"], list)
            else [product["small_description"]]
        )

        pricing = product.get("pricing")
        if pricing is None or not pricing:
            pricing_values = [100.0]
            price_tag = "$100.0"
        else:
            pricing_values = [
                float(Decimal(re.sub(r"[^\d.]", "", price)))
                for price in pricing.split("$")[1:]
            ]
            if len(pricing_values) == 1:
                price_tag = f"${pricing_values[0]}"
            else:
                price_tag = f"${pricing_values[0]} to ${pricing_values[1]}"
                pricing_values = pricing_values[:2]
        item["pricing"] = pricing_values
        item["Price"] = price_tag

        options = {}
        customization_options = product.get("customization_options")
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_values = []
                for option_content in option_contents:
                    option_values.append(
                        option_content["value"].strip().replace("/", " | ").lower()
                    )
                options[option_name.lower()] = option_values
        item["options"] = options

        if asin in attributes and "attributes" in attributes[asin]:
            item["Attributes"] = attributes[asin]["attributes"]
        else:
            item["Attributes"] = ["DUMMY_ATTR"]
        item["instruction_text"] = attributes.get(asin, {}).get("instruction")
        item["instruction_attributes"] = attributes.get(asin, {}).get(
            "instruction_attributes"
        )
        item["MainImage"] = product["images"][0]
        item["query"] = product["query"].lower().strip()
        item["_short_product_order"] = idx
        all_products.append(item)

    product_prices = {}
    for product in all_products:
        pricing = product["pricing"]
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[product["asin"]] = price
    return all_products, product_prices


def _build_synthetic_goals(all_products, product_prices):
    goals = []
    attr_counts = Counter()
    for product in all_products:
        if product.get("instruction_text") is None:
            continue
        asin = product["asin"]
        attributes = product["instruction_attributes"]
        if not attributes:
            continue

        price = product_prices[asin]
        price_range = [p for p in PRICE_RANGE if p > price][:4]
        if len(price_range) >= 2:
            _, price_upper = sorted(random.sample(price_range, 2))
            price_text = f", and price lower than {price_upper:.2f} dollars"
        else:
            price_upper = 1000000
            price_text = ""

        option_names = sorted(product["options"])
        combinations = list(
            itertools.product(*(product["options"][name] for name in option_names))
        )
        for combination in combinations:
            goal_options = {
                option_name: option_value
                for option_name, option_value in zip(option_names, combination)
            }
            option_text = ", and ".join(
                f"{key}: {value}" for key, value in goal_options.items()
            )
            option_text = " with " + option_text if option_text else ""
            goals.append(
                {
                    "asin": asin,
                    "category": product["category"],
                    "query": product["query"],
                    "name": product["Title"],
                    "product_category": product["product_category"],
                    "instruction_text": (
                        f"{product['instruction_text']}{option_text}{price_text}"
                    ),
                    "attributes": attributes,
                    "price_upper": price_upper,
                    "goal_options": goal_options,
                }
            )
            for attribute in attributes:
                attr_counts[attribute] += 1

    for goal in goals:
        goal["weight"] = sum(1.0 / attr_counts[attr] for attr in goal["attributes"]) / len(
            goal["attributes"]
        )
    return goals


def _goals_for_worker_seed(data_dir: Path, worker_seed: int, catalog_random_seed: int):
    all_products, product_prices = _load_short_products(data_dir, catalog_random_seed)
    goals = _build_synthetic_goals(all_products, product_prices)
    random.seed(worker_seed)
    random.shuffle(goals)
    return goals


def main():
    parser = argparse.ArgumentParser(
        description="Simulate standard GRPO WebShop train-goal sampling."
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument(
        "--catalog-random-seed",
        type=int,
        default=0,
        help=(
            "Seed for WebShop price-bound generation. The upstream env does not "
            "seed this before goal construction, so this makes the exported full "
            "goal records deterministic."
        ),
    )
    parser.add_argument(
        "--output",
        default="sft_data/webshop_grpo_samples/grpo_webshop_train_goals_300steps_seed0.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "agent_system/environments/env_package/webshop/webshop/data"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worker_goals = [
        _goals_for_worker_seed(
            data_dir=data_dir,
            worker_seed=args.env_seed + base_env_id,
            catalog_random_seed=args.catalog_random_seed,
        )
        for base_env_id in range(args.train_batch_size)
    ]
    goal_count = len(worker_goals[0])
    goal_idxs = list(range(500, goal_count))
    rng = np.random.RandomState(args.env_seed)

    samples = []
    unique_by_key = {}
    for step in range(args.steps):
        sampled_positions = rng.choice(len(goal_idxs), size=args.train_batch_size, replace=False).tolist()
        sampled_goal_idxs = [goal_idxs[int(pos)] for pos in sampled_positions]
        for base_env_id, (sample_position, goal_idx) in enumerate(
            zip(sampled_positions, sampled_goal_idxs)
        ):
            goal = worker_goals[base_env_id][goal_idx]
            target_key = goal_to_target_key(goal)
            target_key_text = target_key_to_string(target_key)
            entry = {
                "sequence_id": len(samples),
                "step": step,
                "base_env_id": base_env_id,
                "worker_seed": args.env_seed + base_env_id,
                "sample_position_in_train_pool": int(sample_position),
                "goal_idx": int(goal_idx),
                "group_size": args.group_size,
                "replica_env_ids": [
                    base_env_id * args.group_size + replica_id
                    for replica_id in range(args.group_size)
                ],
                "webshop_target_key_text": target_key_text,
                "webshop_target_key": [
                    target_key[0],
                    [[value, count] for value, count in target_key[1]],
                ],
                "instruction_text": goal["instruction_text"],
                "goal": goal,
            }
            samples.append(entry)
            unique_by_key.setdefault(
                target_key_text,
                {
                    "first_sequence_id": entry["sequence_id"],
                    "first_step": step,
                    "first_base_env_id": base_env_id,
                    "webshop_target_key_text": target_key_text,
                    "webshop_target_key": entry["webshop_target_key"],
                    "instruction_text": goal["instruction_text"],
                    "goal": goal,
                    "count_in_ordered_samples": 0,
                },
            )
            unique_by_key[target_key_text]["count_in_ordered_samples"] += 1

    payload = {
        "metadata": {
            "source": "simulated_standard_grpo_webshop_sampling",
            "script": "examples/data_preprocess/collect_webshop_grpo_samples.py",
            "matched_command": "bash examples/grpo_trainer/run_webshop.sh vllm trainer.logger=\"['console','swanlab']\"",
            "steps": args.steps,
            "train_batch_size": args.train_batch_size,
            "group_size": args.group_size,
            "env_seed": args.env_seed,
            "catalog_random_seed": args.catalog_random_seed,
            "goal_pool": "default WebShop train split: goal_idx range(500, len(goals))",
            "base_samples_per_step": args.train_batch_size,
            "rollout_replicas_per_step": args.train_batch_size * args.group_size,
            "ordered_base_sample_count": len(samples),
            "unique_target_key_count": len(unique_by_key),
            "short_goal_count_per_worker": goal_count,
        },
        "ordered_samples": samples,
        "unique_goals": list(unique_by_key.values()),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(output_path)
    print(f"ordered_base_sample_count={len(samples)}")
    print(f"unique_target_key_count={len(unique_by_key)}")


if __name__ == "__main__":
    main()
