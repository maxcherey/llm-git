#!/usr/bin/env python3

import warnings
# Suppress urllib3 warning about OpenSSL
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import subprocess
import sys
import os
from typing import List, Tuple
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.llm_tool import LLMToolBase, parse_args


class LLMGitCommit(LLMToolBase):

    def get_git_diff(self) -> Tuple[str, List[str], str]:
        """Get git diff with context and list of changed files."""
        try:
            diff_output = ""
            patch_output = ""
            changed_files = []

            # Check working directory changes
            diff_cmd = ["git", "diff", "-U30"]
            working_diff = subprocess.check_output(diff_cmd).decode('utf-8')

            patch_cmd = ["git", "diff", "--patch"]
            working_patch = subprocess.check_output(patch_cmd).decode('utf-8')

            files_cmd = ["git", "diff", "--name-only"]
            working_files = subprocess.check_output(files_cmd).decode('utf-8').splitlines()

            # Check staged changes
            staged_diff_cmd = ["git", "diff", "--cached", "-U30"]
            staged_diff = subprocess.check_output(staged_diff_cmd).decode('utf-8')

            staged_patch_cmd = ["git", "diff", "--cached", "--patch"]
            staged_patch = subprocess.check_output(staged_patch_cmd).decode('utf-8')

            staged_files_cmd = ["git", "diff", "--cached", "--name-only"]
            staged_files = subprocess.check_output(staged_files_cmd).decode('utf-8').splitlines()

            # Combine results
            diff_output = working_diff + staged_diff
            patch_output = working_patch + staged_patch
            changed_files = list(set(working_files + staged_files))

            # For new files, get their entire content
            for file in staged_files:
                if file not in working_files:  # This is a newly added file
                    try:
                        with open(file, 'r') as f:
                            content = f.read()
                            diff_output += f"\nNew file: {file}\n{content}\n"
                            patch_output += f"\nNew file: {file}\n{content}\n"
                    except Exception as e:
                        print(f"Warning: Could not read new file {file}: {e}")

            return diff_output, changed_files, patch_output
        except subprocess.CalledProcessError as e:
            print(f"Error getting git diff: {e}")
            sys.exit(1)

    def generate_commit_message(self, diff_output: str) -> str:
        """Generate commit message using LLM API."""
        system_prompt = """
        You are a helpful assistant that analyzes git diffs and generates clear, concise commit messages in plain text.
        The commit message must have a 5-10 words title and reasonable description of the changes below.
        Do not use any prefix or suffix for the title and description like **Title** or **Description**.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a commit message for the following git diff:\n\n{diff_output}"}
        ]

        for endpoint in self.api_endpoints:
            try:
                response = self.make_api_request(endpoint, messages)
                if response:
                    return response.strip()
            except Exception as e:
                print(f"Warning: Failed to use endpoint {endpoint}: {e}")
                continue

        print("Error: All API endpoints failed")
        sys.exit(1)

    def check_untracked_files(self, show_warning=True):
        """Check for untracked files that aren't ignored."""
        try:
            # Get list of untracked files that aren't ignored
            cmd = ["git", "ls-files", "--others", "--exclude-standard"]
            untracked = subprocess.check_output(cmd).decode('utf-8').splitlines()

            if untracked and show_warning:
                self.show_untracked_warning(untracked)

            return untracked
        except subprocess.CalledProcessError as e:
            print(f"Error checking untracked files: {e}")
            return []

    def show_untracked_warning(self, untracked):
        """Show warning about untracked files."""
        print("="*80)
        print("Warning: The following files are not tracked by git:")
        for file in untracked:
            print(f"  - {file}")
        print("\nUse 'git add' to track or add them to .gitignore to ignore")
        print("="*80)

    def run(self):
        """Main execution flow."""
        try:
            # Get untracked files but don't show warning yet
            untracked = self.check_untracked_files(show_warning=False)

            # Discover available endpoints and models
            available_endpoints = self.discover_endpoints()
            self.api_endpoints = available_endpoints

            logging.info(f"Using model: {self.model}")

            # Get git diff and changed files
            diff_output, changed_files, patch_output = self.get_git_diff()

            if not diff_output and not changed_files:  # Changed condition
                print("\n" + "="*80)
                print("No changes in git working directory, nothing to commit")
                print("="*80 + "\n")
                # Show untracked files warning if any
                if untracked:
                    self.show_untracked_warning(untracked)
                return

            # Generate commit message
            commit_message = self.generate_commit_message(diff_output)

            # Section 1: Show changes
            print("\n" + "="*80)
            if self.quiet:
                print("Changed files:")
                for file in changed_files:
                    print(f"  - {file}")
            else:
                print("Changes to be committed:")
                print("-"*80)
                print(patch_output.rstrip())
            print("="*80)

            # Section 2: Show model info and proposed commit
            print(f"Using LLM Model: {self.model}, Proposed commit command:")
            print()
            commit_cmd = f'git commit -m "{commit_message}" {" ".join(changed_files)}'
            print(commit_cmd)
            print("="*80)

            # Ask for confirmation with N as default
            response = input("\nProceed with commit? [y/N]: ")

            # Section 3: Show final result
            print("\n" + "="*80)
            if response.lower() != 'y':
                print("Commit cancelled")
            else:
                try:
                    subprocess.run(["git", "commit", "-m", commit_message] + changed_files, check=True)
                    print(f"✓ Committed {len(changed_files)} file(s)")
                except subprocess.CalledProcessError as e:
                    print(f"Error during commit: {e}")
                    sys.exit(1)
            print("="*80)

            # Show untracked files warning if any
            if untracked:
                print()  # Add empty line for spacing
                self.show_untracked_warning(untracked)

        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            sys.exit(1)


def main():
    args = parse_args()

    api_endpoints = [args.api_url] if args.api_url else None
    helper = LLMGitCommit(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_endpoints=api_endpoints,
        model=args.model,
        verbose=args.verbose,
        quiet=args.quiet
    )
    helper.run()


if __name__ == "__main__":
    main()
