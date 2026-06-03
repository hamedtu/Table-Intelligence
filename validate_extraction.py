"""Validation script to analyze and verify table cell extraction results. 

This script processes the output from extract_cells.py, checks for completeness and accuracy of cell extraction,
and generates a detailed report. 
The script is designed to be run after the extraction process to ensure that the results meet quality standards and 
to guide any necessary adjustments in extraction parameters.


Authorship Note:
This script was developed with AI code assistance (GitHub Copilot) for initial scaffolding and suggestions,
but all core design decisions, logic flow, error handling, and custom validations were manually reviewed,
tested, and refined by the author [Hamed Kelardeh].

""" 

import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def analyze_results(results_dir: Path) -> Dict[str, dict]:
    """Analyze extraction results for all processed tables.
    
    Key design decisions:
    - Assumes cell naming convention: cell-{row}-{col}.png (zero-indexed)
    - Uses max_row/max_col to infer grid dimensions (assumes complete grid structure)
    - Coverage calculation: detected_cells / (max_row+1)*(max_col+1) * 100
    
    Args:
        results_dir: Directory containing extraction results (one subdirectory per table)
        
    Returns:
        Dictionary mapping table name to analysis metrics
    """
    analysis = {}
    
    # Find all table subdirectories - expects structure: results_dir/table_name/cell-*.png
    table_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    for table_dir in sorted(table_dirs):
        table_name = table_dir.name
        
        # Find all extracted cell images - filename pattern is critical for parsing
        cell_files = list(table_dir.glob("cell-*.png"))
        
        # Handle empty tables gracefully (no cells extracted)
        if len(cell_files) == 0:
            analysis[table_name] = {
                'total_cells': 0,
                'max_row': 0,
                'max_col': 0,
                'grid_size': (0, 0),
                'status': 'EMPTY'
            }
            continue
        
        # Parse row and column indices from filenames using regex
        # Pattern: cell-{row}-{col}.png where row and col are integers
        rows = []
        cols = []
        cell_pattern = re.compile(r'cell-(\d+)-(\d+)\.png')
        
        for cell_file in cell_files:
            match = cell_pattern.match(cell_file.name)
            if match:
                # Extract row and column indices (zero-indexed)
                row = int(match.group(1))
                col = int(match.group(2))
                rows.append(row)
                cols.append(col)
        
        # Calculate grid metrics from extracted indices
        # Note: Assumes 0-indexed rows/cols, so actual grid size is max_idx + 1
        max_row = max(rows) if rows else 0
        max_col = max(cols) if cols else 0
        
        # Calculate expected cells for a complete grid
        # Assumption: if max_row=2 and max_col=3, we expect a 3x4 grid (complete grid structure)
        expected_cells = (max_row + 1) * (max_col + 1)
        
        # Coverage percentage: actual_cells / expected_cells
        # This metric helps identify sparse or incomplete extractions
        coverage = len(cell_files) / expected_cells * 100 if expected_cells > 0 else 0
        
        # Determine quality status based on coverage threshold
        # These thresholds were chosen to provide meaningful categorization:
        # - EXCELLENT: >95% suggests nearly complete extraction
        # - GOOD: 80-95% suggests minor missing cells
        # - FAIR: 60-80% suggests significant gaps but usable
        # - POOR: <60% suggests major issues needing investigation
        if coverage >= 95:
            status = 'EXCELLENT'
        elif coverage >= 80:
            status = 'GOOD'
        elif coverage >= 60:
            status = 'FAIR'
        else:
            status = 'POOR'
        
        analysis[table_name] = {
            'total_cells': len(cell_files),
            'max_row': max_row,
            'max_col': max_col,
            'grid_size': (max_row + 1, max_col + 1),
            'expected_cells': expected_cells,
            'coverage': coverage,
            'status': status
        }
    
    return analysis


def print_analysis(analysis: Dict[str, dict]):
    """Print analysis results in a formatted table with summary statistics.
    
    Output structure:
    1. Per-table detailed view: Each table's extraction metrics for quick scanning
    2. Summary statistics: Aggregate metrics across all tables
    
    This structure allows users to:
    - Quickly identify problematic tables (scanning Status column)
    - Compare relative performance across tables (Coverage column)
    - Understand overall extraction quality at a glance (Summary section)
    
    Args:
        analysis: Analysis results from analyze_results()
    """
    print("\n" + "="*80)
    print("TABLE CELL EXTRACTION ANALYSIS REPORT")
    print("="*80)
    
    # Print detailed per-table results
    # Rationale for column order: table name first (identification), then metrics that matter most
    # (cells extracted, grid size), then performance metrics (coverage, status)
    print("\nPer-Table Results:")
    print("-" * 80)
    print(f"{'Table':<15} {'Cells':<8} {'Grid':<12} {'Expected':<10} {'Coverage':<10} {'Status':<10}")
    print("-" * 80)
    
    # Sort by table name for consistent, reproducible output
    for table_name in sorted(analysis.keys()):
        metrics = analysis[table_name]
        grid_str = f"{metrics['grid_size'][0]}x{metrics['grid_size'][1]}"
        # Handle division by zero: show N/A if no cells extracted
        coverage_str = f"{metrics['coverage']:.1f}%" if metrics['total_cells'] > 0 else "N/A"
        
        print(f"{table_name:<15} {metrics['total_cells']:<8} {grid_str:<12} "
              f"{metrics['expected_cells']:<10} {coverage_str:<10} {metrics['status']:<10}")
    
    # Print summary statistics
    # These aggregate metrics provide high-level insights into overall extraction performance
    print("\n" + "-" * 80)
    print("Summary Statistics:")
    print("-" * 80)
    
    total_tables = len(analysis)
    # Sum cells across all tables
    total_cells = sum(m['total_cells'] for m in analysis.values())
    
    # Count status distribution - helps understand overall quality
    status_counts = defaultdict(int)
    for metrics in analysis.values():
        status_counts[metrics['status']] += 1
    
    # Calculate coverage statistics (only for non-empty tables)
    # Excluding EMPTY tables from coverage average ensures meaningful average
    coverages = [m['coverage'] for m in analysis.values() if m['total_cells'] > 0]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0
    min_coverage = min(coverages) if coverages else 0
    max_coverage = max(coverages) if coverages else 0
    
    print(f"Total tables processed: {total_tables}")
    print(f"Total cells extracted: {total_cells}")
    print(f"Average coverage: {avg_coverage:.1f}%")
    print(f"Coverage range: {min_coverage:.1f}% - {max_coverage:.1f}%")
    print()
    print("Status Distribution:")
    # Show status distribution only for statuses that exist
    for status in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'EMPTY']:
        count = status_counts[status]
        if count > 0:
            pct = count / total_tables * 100
            print(f"  {status}: {count} tables ({pct:.1f}%)")
    
    print("\n" + "="*80)


def identify_issues(analysis: Dict[str, dict], threshold: float = 80.0) -> List[str]:
    """Identify tables with extraction issues based on coverage threshold.
    
    Design rationale:
    - Default threshold of 80% was chosen as a balance between strictness and practical usability
    - Made configurable to support different quality standards
    - Two issue types: no cells extracted (structural problem) vs low coverage (partial problem)
    
    Args:
        analysis: Analysis results from analyze_results()
        threshold: Minimum coverage percentage for acceptable extraction (default: 80%)
        
    Returns:
        List of table names with issues, formatted with specific problem descriptions
    """
    issues = []
    
    for table_name, metrics in analysis.items():
        # Check for structural failures: no cells extracted at all
        if metrics['total_cells'] == 0:
            issues.append(f"{table_name}: No cells extracted")
        # Check for incomplete extraction: coverage below acceptable threshold
        elif metrics['coverage'] < threshold:
            issues.append(f"{table_name}: Low coverage ({metrics['coverage']:.1f}%)")
    
    return issues


def validate_grid_continuity(analysis: Dict[str, dict]) -> Dict[str, bool]:
    """Validate that extracted cells form a continuous grid (no sparse gaps).
    
    Additional validation I added: This function checks if extracted cells might indicate
    systematic extraction failures (e.g., pattern of missing rows or columns).
    
    Args:
        analysis: Analysis results from analyze_results()
        
    Returns:
        Dictionary mapping table name to whether grid appears continuous
        (Note: This is a heuristic - actual cell validation would require file inspection)
    """
    continuity_check = {}
    
    for table_name, metrics in analysis.items():
        # If coverage is very close to 100%, grid is likely continuous
        # Otherwise, sparse grids might indicate systematic issues
        is_continuous = metrics['coverage'] >= 95.0 or metrics['total_cells'] == 0
        continuity_check[table_name] = is_continuous
    
    return continuity_check


def main():
    """Main entry point for validation script.
    
    Workflow:
    1. Parse command line arguments (results directory, threshold)
    2. Validate that results directory exists
    3. Analyze all tables in the directory
    4. Print formatted report with statistics
    5. Identify and report problematic tables
    
    Exit codes:
    - 0: Success (with or without issues found)
    - 1: Error (missing directory or no tables found)
    """
    parser = argparse.ArgumentParser(
        description="Validate and analyze table cell extraction results",
        epilog="""
Examples:
  # Validate default results directory
  python validate_extraction.py results
  
  # Validate custom directory with custom threshold
  python validate_extraction.py /path/to/results
        """
    )
    parser.add_argument(
        "results_dir",
        type=str,
        nargs="?",
        default="./results",
        help="Directory containing extraction results (default: ./results)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Coverage threshold for issue detection (default: 80.0)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Validate input: ensure results directory exists before processing
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return 1
    
    # Analyze results and generate metrics
    analysis = analyze_results(results_dir)
    
    # Check if any tables were found: empty directory is an error condition
    if len(analysis) == 0:
        print(f"No tables found in '{results_dir}'")
        return 1
    
    # Generate and print detailed analysis report
    print_analysis(analysis)
    
    # Identify and report tables with issues
    issues = identify_issues(analysis, args.threshold)
    
    # Provide actionable feedback based on whether issues were found
    if issues:
        print("\n" + "="*80)
        print(f"TABLES WITH ISSUES (Coverage < {args.threshold}%):")
        print("="*80)
        for issue in issues:
            print(f"  • {issue}")
        print("\nSuggestion: Consider adjusting extraction parameters for these tables.")
        print("See TableConfig in extract_cells.py for tunable parameters.")
    else:
        print("\n✓ All tables meet the coverage threshold!")
    
    return 0


if __name__ == "__main__":
    exit(main())
