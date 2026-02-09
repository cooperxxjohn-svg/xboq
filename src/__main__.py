"""
Floor Plan Room & Area Engine - CLI Entry Point

Commands:
    process   - Process single floor plan
    project   - Process multi-page PDF project (sheet-aware)
    classify  - Classify pages in a PDF
    scale     - Infer scale from dimensions
    test      - Run tests
"""

import argparse
import logging
import sys
from pathlib import Path

from .floorplan_pipeline import FloorPlanPipeline, PipelineConfig, process_plan, process_batch


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def cmd_process(args):
    """Process floor plan(s)."""
    input_path = Path(args.input)
    output_dir = Path(args.output)

    config = PipelineConfig(
        dpi=args.dpi,
        default_scale=args.scale,
        output_dir=output_dir
    )

    if input_path.is_file():
        # Single file
        result = process_plan(input_path, output_dir, config)
        print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Rooms: {len(result.rooms_with_area)}")
        if result.qc_report:
            print(f"Total area: {result.qc_report.statistics.get('total_area_sqm', 0):.1f} sqm")
            print(f"Confidence: {result.qc_report.overall_confidence:.0%}")
        if result.output_paths:
            print(f"Outputs: {output_dir / result.plan_id}")

    elif input_path.is_dir():
        # Directory - process all plans
        files = list(input_path.glob("*.pdf")) + \
                list(input_path.glob("*.png")) + \
                list(input_path.glob("*.jpg"))

        if not files:
            print(f"No floor plan files found in {input_path}")
            return 1

        results = process_batch(files, output_dir, config)

        # Summary
        success = len([r for r in results if r.success])
        print(f"\nProcessed: {success}/{len(results)} successful")

    else:
        print(f"Input not found: {input_path}")
        return 1

    return 0


def cmd_export(args):
    """Export a single plan."""
    from .ingest import ingest_plan
    from .export import PlanExporter

    input_path = Path(args.plan)
    output_dir = Path(args.output)

    config = PipelineConfig(
        dpi=args.dpi,
        default_scale=args.scale,
        output_dir=output_dir
    )

    pipeline = FloorPlanPipeline(config)
    result = pipeline.process(input_path)

    if result.success:
        print(f"Exported to: {output_dir / result.plan_id}")
    else:
        print(f"Export failed: {result.errors}")
        return 1

    return 0


def cmd_test(args):
    """Run tests."""
    from tests.test_pipeline import run_all_tests

    success = run_all_tests()
    return 0 if success else 1


def cmd_project(args):
    """Process multi-page PDF project with sheet classification."""
    from .project_pipeline import process_project, ProjectConfig

    input_path = Path(args.input)
    output_dir = Path(args.output)

    config = ProjectConfig(
        dpi=args.dpi,
        default_scale=args.scale,
        enable_dimension_scale=not args.no_dimension_scale,
        save_debug_images=args.debug,
    )

    print(f"Processing project: {input_path}")
    result = process_project(input_path, output_dir, config)

    print(f"\n{'='*60}")
    print(f"PROJECT RESULT: {result.project_id}")
    print(f"{'='*60}")
    print(f"Total pages: {result.total_pages}")
    print(f"Processed: {result.pages_processed}")
    print(f"Skipped: {result.pages_skipped}")
    print(f"Total rooms: {result.total_rooms}")
    print(f"Total area: {result.total_area_sqm:.1f} sqm ({result.total_area_sqm * 10.764:.1f} sqft)")
    print(f"Processing time: {result.processing_time_sec:.1f}s")
    print(f"\nOutputs: {output_dir / result.project_id}/")
    print(f"  - manifest.json")
    print(f"  - summary.md")
    print(f"  - results.json")
    if args.debug:
        print(f"  - debug/ (classification and scale images)")

    return 0


def cmd_classify(args):
    """Classify pages in a PDF."""
    from .multipage import ingest_multipage_project

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    pages, manifest = ingest_multipage_project(input_path, output_dir, dpi=args.dpi)

    print(f"\nPage Classification: {manifest.project_id}")
    print(f"Total pages: {manifest.total_pages}")
    print()
    print(f"{'Page':<6} {'Type':<20} {'Conf':<8} {'Process':<10} {'Title':<30}")
    print("-" * 80)

    for page_data, classification in pages:
        process = "Yes" if classification.should_process_rooms else "No"
        title = classification.title_block.get("title", "")[:28]
        print(f"{page_data.page_number + 1:<6} {classification.sheet_type.value:<20} "
              f"{classification.confidence:<8.0%} {process:<10} {title:<30}")

    print()
    print("Summary:")
    for stype, count in manifest.summary.get("sheet_types", {}).items():
        print(f"  {stype}: {count}")

    if output_dir:
        manifest_path = output_dir / manifest.project_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(manifest_path)
        print(f"\nManifest saved to: {manifest_path}")

    return 0


def cmd_scale(args):
    """Infer scale from dimensions."""
    from .scale_dimensions import infer_scale_from_dimensions
    import cv2

    input_path = Path(args.input)
    img = cv2.imread(str(input_path))

    if img is None:
        print(f"Could not read image: {input_path}")
        return 1

    debug_path = None
    if args.debug:
        debug_path = Path(args.output) / "scale_debug.png"
        debug_path.parent.mkdir(parents=True, exist_ok=True)

    result = infer_scale_from_dimensions(img, debug_path=debug_path)

    print(f"\nScale Inference: {input_path.name}")
    print(f"{'='*50}")
    print(f"Method: {result.method}")
    print(f"Pixels per mm: {result.pixels_per_mm:.4f}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Dimensions used: {result.num_dimensions_used}")

    if result.inliers:
        print(f"\nInlier dimensions:")
        for m in result.inliers[:5]:
            print(f"  {m.candidate.raw_text}: {m.pixels_per_mm:.4f} px/mm")

    if result.outliers:
        print(f"\nOutlier dimensions (excluded):")
        for m in result.outliers[:3]:
            print(f"  {m.candidate.raw_text}: {m.pixels_per_mm:.4f} px/mm")

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if debug_path:
        print(f"\nDebug image: {debug_path}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Floor Plan Room & Area Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single plan
  python -m src --input plan.pdf --output ./out

  # Process all plans in a directory
  python -m src --input ./data/plans --output ./out

  # Run tests
  python -m src test

  # Start UI
  streamlit run app/app.py
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Process command
    proc_parser = subparsers.add_parser('process', help='Process floor plan(s)')
    proc_parser.add_argument('--input', '-i', required=True,
                            help='Input file or directory')
    proc_parser.add_argument('--output', '-o', default='./out',
                            help='Output directory')
    proc_parser.add_argument('--dpi', type=int, default=300,
                            help='DPI for rendering (default: 300)')
    proc_parser.add_argument('--scale', type=int, default=100,
                            help='Default scale ratio (default: 100)')
    proc_parser.set_defaults(func=cmd_process)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export single plan')
    export_parser.add_argument('--plan', '-p', required=True,
                              help='Plan file to export')
    export_parser.add_argument('--output', '-o', default='./out',
                              help='Output directory')
    export_parser.add_argument('--dpi', type=int, default=300,
                              help='DPI for rendering')
    export_parser.add_argument('--scale', type=int, default=100,
                              help='Default scale ratio')
    export_parser.set_defaults(func=cmd_export)

    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.set_defaults(func=cmd_test)

    # Project command (multi-page)
    project_parser = subparsers.add_parser('project',
                                           help='Process multi-page PDF project')
    project_parser.add_argument('--input', '-i', required=True,
                                help='Input PDF file')
    project_parser.add_argument('--output', '-o', default='./out',
                                help='Output directory')
    project_parser.add_argument('--dpi', type=int, default=300,
                                help='DPI for rendering')
    project_parser.add_argument('--scale', type=int, default=100,
                                help='Default scale ratio')
    project_parser.add_argument('--no-dimension-scale', action='store_true',
                                help='Disable dimension-based scale inference')
    project_parser.add_argument('--debug', action='store_true',
                                help='Save debug images')
    project_parser.set_defaults(func=cmd_project)

    # Classify command
    classify_parser = subparsers.add_parser('classify',
                                            help='Classify pages in a PDF')
    classify_parser.add_argument('--input', '-i', required=True,
                                 help='Input PDF file')
    classify_parser.add_argument('--output', '-o',
                                 help='Output directory for manifest')
    classify_parser.add_argument('--dpi', type=int, default=200,
                                 help='DPI for rendering (lower for speed)')
    classify_parser.set_defaults(func=cmd_classify)

    # Scale command
    scale_parser = subparsers.add_parser('scale',
                                         help='Infer scale from dimensions')
    scale_parser.add_argument('--input', '-i', required=True,
                              help='Input image file')
    scale_parser.add_argument('--output', '-o', default='./out',
                              help='Output directory for debug image')
    scale_parser.add_argument('--debug', action='store_true',
                              help='Save debug visualization')
    scale_parser.set_defaults(func=cmd_scale)

    # Also allow direct process without subcommand
    parser.add_argument('--input', '-i', help='Input file or directory')
    parser.add_argument('--output', '-o', default='./out', help='Output directory')
    parser.add_argument('--dpi', type=int, default=300, help='DPI')
    parser.add_argument('--scale', type=int, default=100, help='Default scale')

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command:
        return args.func(args)
    elif args.input:
        # Direct process mode
        args.func = cmd_process
        return cmd_process(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
