#!/usr/bin/env python3
"""Small test harness to exercise the fraud flow without LiveKit dependencies.

Usage:
  python tools/test_fraud_flow.py --username "John Doe" --verify-answer "blue" --made-transaction false

This will load `shared-data/fraud_cases.json`, find the case for the username,
check the verification answer, print the suspicious transaction, and optionally
resolve the case (mark safe or fraudulent) and save the DB.
"""
import argparse
import json
import os
import sys


def load_db(path):
    if not os.path.exists(path):
        print(f"Fraud DB not found: {path}")
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def find_case(data, username):
    uname = (username or "").lower()
    for i, entry in enumerate(data):
        if (entry.get("userName") or "").lower() == uname:
            return i, entry
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--username", required=True)
    p.add_argument("--verify-answer", default=None)
    p.add_argument("--made-transaction", choices=["true", "false"], default=None)
    args = p.parse_args()

    db_path = os.path.join(os.getcwd(), "shared-data", "fraud_cases.json")
    data = load_db(db_path)
    idx, case = find_case(data, args.username)
    if case is None:
        print(f"No case found for user '{args.username}'")
        sys.exit(1)

    print("Loaded case:")
    for k in ["userName", "transactionName", "transactionAmount", "transactionTime", "cardEnding", "location", "transactionCategory"]:
        print(f"  {k}: {case.get(k)}")

    if args.verify_answer is not None:
        expected = (case.get("securityAnswer") or "").strip().lower()
        got = args.verify_answer.strip().lower()
        if expected and got == expected:
            print("Verification: VERIFIED")
        else:
            print("Verification: FAILED")
            case["status"] = "verification_failed"
            case["outcome_note"] = "Verification failed during fraud call."
            data[idx] = case
            save_db_atomic(db_path, data)
            print("Updated DB with verification_failed status.")
            sys.exit(0)

    if args.made_transaction is not None:
        made = args.made_transaction == "true"
        if made:
            case["status"] = "confirmed_safe"
            case["outcome_note"] = "Customer confirmed transaction as legitimate."
            print("Outcome: confirmed_safe")
        else:
            case["status"] = "confirmed_fraud"
            case["outcome_note"] = "Customer reported transaction as fraudulent. Card blocked and dispute initiated (mock)."
            print("Outcome: confirmed_fraud")
        data[idx] = case
        save_db_atomic(db_path, data)
        print("Saved updated case to DB.")
        sys.exit(0)

    print('\nNo action taken. To verify, pass --verify-answer. To resolve, pass --made-transaction true|false')


if __name__ == "__main__":
    main()
