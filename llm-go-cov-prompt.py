#!/usr/bin/env python3

import warnings
# Suppress urllib3 warning about OpenSSL
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import subprocess
import sys
import argparse
import os
import json
import requests
from typing import List, Tuple, Dict
import logging
import re
import fnmatch


class LLMGoCovPrompt:
    def __init__(self, coverage_out_file="coverage.out", files=None,
                 temperature=0.5, max_tokens=16384, api_endpoints=None,
                 model=None, verbose=0, quiet=False, threshold=None):
        self.coverage_out_file = coverage_out_file
        self.files = files or ["."]  # Default to current directory if not specified
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_endpoints = api_endpoints
        self.model = model
        self.verbose = verbose
        self.quiet = quiet
        self.threshold = threshold  # Coverage threshold percentage to ignore files

        # Configure logging
        setup_logging(verbose)

    def parse_coverage_file(self) -> Tuple[Dict[str, List[Tuple[int, int]]], str]:
        """
        Parse the coverage.out file to extract uncovered code sections.
        Merges sections that intersect or are adjacent.

        Returns:
            Tuple containing:
            - Dictionary mapping file paths to list of (start, end) line ranges
            - Common prefix detected in file paths
        """
        uncovered_sections = {}
        all_files = []

        logging.info(f"Parsing coverage file: {self.coverage_out_file}")

        try:
            with open(self.coverage_out_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines or lines that don't match the pattern
                    if not line or ':' not in line:
                        continue

                    # The last token being 0 indicates no coverage
                    parts = line.split()
                    if len(parts) < 3 or parts[-1] != '0':
                        continue

                    # Extract file path and line range
                    file_info = parts[0]
                    file_path, line_range = file_info.split(':')
                    all_files.append(file_path)

                    # Extract start and end line numbers
                    line_parts = line_range.split(',')
                    if len(line_parts) < 2:
                        continue

                    start_line = int(line_parts[0].split('.')[0])
                    end_line = int(line_parts[1].split('.')[0])

                    # Add to uncovered sections
                    if file_path not in uncovered_sections:
                        uncovered_sections[file_path] = []

                    uncovered_sections[file_path].append((start_line, end_line))

            # Merge overlapping or adjacent sections for each file
            for file_path in uncovered_sections:
                uncovered_sections[file_path] = self._merge_overlapping_sections(uncovered_sections[file_path])
                logging.info(f"File {file_path}: {len(uncovered_sections[file_path])} merged sections")

            # Detect common prefix
            common_prefix = self.detect_common_prefix(all_files)
            logging.info(f"Detected common prefix: {common_prefix}")

            return uncovered_sections, common_prefix

        except FileNotFoundError:
            logging.error(f"Coverage file not found: {self.coverage_out_file}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error parsing coverage file: {str(e)}")
            sys.exit(1)

    def _merge_overlapping_sections(self, sections: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge sections that overlap or are adjacent.

        Args:
            sections: List of (start, end) tuples representing line ranges

        Returns:
            List of merged (start, end) tuples
        """
        if not sections:
            return []

        # Sort sections by start line
        sorted_sections = sorted(sections, key=lambda x: x[0])

        # Initialize with the first section
        merged = [sorted_sections[0]]

        for current in sorted_sections[1:]:
            prev = merged[-1]

            # Check if current section overlaps or is adjacent to the previous one
            # Two sections (prev_start, prev_end) and (curr_start, curr_end) are:
            # - Overlapping if: prev_start <= curr_end and curr_start <= prev_end
            # - Adjacent if: prev_end + 1 == curr_start or curr_end + 1 == prev_start

            if (prev[0] <= current[1] and current[0] <= prev[1]) or \
               prev[1] + 1 == current[0] or current[1] + 1 == prev[0]:
                # Merge by taking the min start and max end
                merged[-1] = (min(prev[0], current[0]), max(prev[1], current[1]))
            else:
                # If not overlapping or adjacent, add as a new section
                merged.append(current)

        logging.debug(f"Merged {len(sections)} sections into {len(merged)} sections")
        return merged

    def detect_common_prefix(self, file_paths: List[str]) -> str:
        """
        Detect common prefix in file paths (likely the Go module path).

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Common prefix string
        """
        if not file_paths:
            return ""

        # Find common prefix
        prefix = os.path.commonprefix(file_paths)

        # Ensure the prefix ends at a module boundary (ends with /)
        if not prefix.endswith('/'):
            last_slash = prefix.rfind('/')
            if last_slash > 0:
                prefix = prefix[:last_slash + 1]

        return prefix

    def should_process_file(self, file_path: str) -> bool:
        """
        Check if a file should be processed based on the provided files/directories.
        Supports wildcard patterns like '*', 'abc*', etc.

        Args:
            file_path: File path to check

        Returns:
            True if the file should be processed, False otherwise
        """
        # Convert to local file path
        local_path = file_path
        logging.info(f"Checking if file {local_path} should be processed")

        # Check if file is within any of the specified directories or matches patterns
        for target in self.files:
            if local_path.startswith(target) or target == ".":
                logging.debug(f"File matches prefix: {target}")
                return True

        logging.debug(f"Skipping file {local_path} (not in target files/directories)")
        return False

    def annotate_file(self, file_path: str, ranges: List[Tuple[int, int]]) -> str:
        """
        Annotate a file with comments indicating where unit tests are needed.

        Args:
            file_path: Path to the file to annotate
            ranges: List of (start, end) line ranges that need tests

        Returns:
            Annotated file content as a string
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            result = []
            line_idx = 0

            # Sort ranges to process them in order
            ranges.sort()

            for line_idx, line in enumerate(lines):
                current_line_num = line_idx + 1  # 1-based line numbers

                # Check if this line is the start of an uncovered section
                for start, end in ranges:
                    if current_line_num == start + 1:
                        result.append("// >>> GENERATE MORE UNIT TESTS FOR THIS SECTION BELOW TO INCRASE CODE COVERAGE:\n")

                # Add the original line
                result.append(line)

                # Check if this line is the end of an uncovered section
                for start, end in ranges:
                    if current_line_num == end - 1:
                        result.append("// <<< END OF SECTION WHERE WE NEED MORE UNIT TESTS FOR BETTER COVERAGE\n")

            return ''.join(result)

        except FileNotFoundError:
            logging.warning(f"File not found: {file_path}")
            return None
        except Exception as e:
            logging.warning(f"Error annotating file {file_path}: {str(e)}")
            return None

    def run(self):
        """Main execution method to process coverage and annotate files."""
        # Parse coverage file
        uncovered_sections, common_prefix = self.parse_coverage_file()

        if not uncovered_sections:
            logging.info("No uncovered sections found in the coverage file.")
            return

        # Process each file
        for file_path, ranges in uncovered_sections.items():
            # Remove common prefix to get local file path
            local_path = file_path
            if file_path.startswith(common_prefix):
                local_path = file_path[len(common_prefix):]

            # Check if file should be processed based on specified files/directories
            if not self.should_process_file(local_path):
                continue

            logging.info(f"Processing file: {local_path}")

            # Calculate coverage statistics
            try:
                with open(local_path, 'r') as f:
                    total_lines = sum(1 for _ in f)

                # Count uncovered lines from ranges
                uncovered_lines = 0
                for start, end in ranges:
                    uncovered_lines += (end - start)

                # Calculate coverage percentage
                if total_lines > 0:
                    coverage_percent = 100 * (total_lines - uncovered_lines) / total_lines
                else:
                    coverage_percent = 100.0

                logging.info(f"File {local_path}: {coverage_percent:.2f}% coverage ({uncovered_lines} uncovered lines)")

                # Skip files with coverage percentage above threshold
                if self.threshold is not None and coverage_percent > self.threshold:
                    logging.info(f"Skipping file {local_path}: coverage {coverage_percent:.2f}% exceeds threshold {self.threshold}%")
                    continue

            except Exception as e:
                logging.warning(f"Error calculating coverage for {local_path}: {str(e)}")
                coverage_percent = None
                uncovered_lines = sum(end - start for start, end in ranges)

            # Annotate the file
            annotated_content = self.annotate_file(local_path, ranges)
            if annotated_content:
                # Print the annotated content
                print("="*80)
                print(f"Update unit tests for file: {local_path}")

                # Print coverage information
                if coverage_percent is not None:
                    print(f"Coverage: {coverage_percent:.2f}% ({uncovered_lines} uncovered lines, {len(ranges)} sections)")
                else:
                    print(f"Uncovered: {uncovered_lines} lines in {len(ranges)} sections")

                print(f"File: {local_path}")
                print("-"*80)
                print(annotated_content)
                print("="*80)
                print()


def setup_logging(verbose: int):
    """Configure logging based on verbosity level."""
    if verbose >= 2:
        log_level = logging.DEBUG
    elif verbose >= 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description="Git commit helper with LLM-generated commit messages")

    # Create coverage options group
    coverage_group = parser.add_argument_group('Coverage Options')
    coverage_group.add_argument(
        "-c", "--coverage-out-file",
        type=str,
        default="coverage.out",
        help="Path to the coverage output file (default: coverage.out)"
    )
    coverage_group.add_argument(
        "-t", "--threshold",
        type=float,
        help="Skip files with coverage percentage above this threshold"
    )

    # Create output control group
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (-v for INFO, -vv for DEBUG)"
    )
    output_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - don't show patch"
    )

    # Add files/directories argument
    parser.add_argument(
        "files",
        nargs="*",
        help="Files or directories to process (default: current directory)"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logging.info("Using files: %s", args.files)

    # Create LLMGoCovPrompt instance
    helper = LLMGoCovPrompt(
        coverage_out_file=args.coverage_out_file,
        files=args.files if args.files else ["."],
        verbose=args.verbose,
        quiet=args.quiet,
        threshold=args.threshold
    )

    # Run the process
    helper.run()


if __name__ == "__main__":
    main()
