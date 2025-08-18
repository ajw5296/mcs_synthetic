import argparse
import pandas as pd
from pathlib import Path
import logging
import json

from analyzer import DatasetAnalyzer


def main():
    """Analyze a generated synthetic dataset"""
    
    parser = argparse.ArgumentParser(description="Analyze synthetic maltreatment dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output-dir", type=str, default="./analysis", help="Output directory for analysis results")
    parser.add_argument("--visualizations", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from: {args.input}")
        
        input_path = Path(args.input)
        if input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        elif input_path.suffix == '.json':
            df = pd.read_json(input_path)
        elif input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzer
        analyzer = DatasetAnalyzer(df)
        
        # Generate summary report
        logger.info("Generating summary report...")
        report_path = output_dir / "analysis_report.json"
        report = analyzer.generate_summary_report(report_path)
        
        # Print key statistics
        print("\n" + "="*50)
        print("DATASET ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"\nDataset Overview:")
        print(f"  Total Cases: {report['dataset_overview']['total_cases']}")
        print(f"  Total Columns: {report['dataset_overview']['total_columns']}")
        
        print(f"\nBinary Coding Summary:")
        for abuse_type, stats in report['binary_coding_analysis'].items():
            print(f"  {abuse_type}: {stats['true_count']} cases ({stats['true_percentage']:.1f}%)")
        
        print(f"\nData Quality Checks:")
        total_inconsistent = 0
        for check_name, check_data in report['data_quality_checks']['consistency_checks'].items():
            inconsistent = check_data['total_inconsistent']
            total_inconsistent += inconsistent
            if inconsistent > 0:
                print(f"  {check_name}: {inconsistent} inconsistent cases")
        
        if total_inconsistent == 0:
            print("  ✓ All consistency checks passed")
        
        validation_errors = report['data_quality_checks']['validation_errors']
        if validation_errors:
            print(f"\nValidation Errors:")
            for error in validation_errors:
                print(f"  ⚠ {error}")
        else:
            print("  ✓ No validation errors found")
        
        # Generate visualizations if requested
        if args.visualizations:
            logger.info("Generating visualizations...")
            viz_dir = output_dir / "visualizations"
            analyzer.create_visualizations(viz_dir)
        
        logger.info(f"Analysis completed. Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
