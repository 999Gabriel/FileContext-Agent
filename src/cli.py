#!/usr/bin/env python3
import sys
import argparse
import json
from agent import SmartFileContextAgent


def main():
    parser = argparse.ArgumentParser(description="Smart File Context Agent")
    parser.add_argument("command", choices=["ask", "status", "start"], help="Command to execute")
    parser.add_argument("--question", "-q", help="Question to ask (for 'ask' command)")
    parser.add_argument("--folder", "-f", help="Project folder to watch", default=".")

    args = parser.parse_args()

    agent = SmartFileContextAgent()

    if args.command == "start":
        try:
            agent.start(args.folder)
            print("Press Ctrl+C to stop...")
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            agent.stop()

    elif args.command == "ask":
        if not args.question:
            print("Please provide a question with --question or -q")
            sys.exit(1)

        # Start agent without file watching for one-time query
        agent.file_watcher.scan_and_index_all(args.folder)

        result = agent.ask(args.question)
        print(json.dumps(result, indent=2))

    elif args.command == "status":
        agent.file_watcher.scan_and_index_all(args.folder)
        status = agent.status()
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()