import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
from typing import Optional

from models import GenerationConfig
from data_generator import SyntheticDataGenerator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_generation_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: Optional[str] = None) -> GenerationConfig:
    """Load configuration from file or environment variables"""
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return GenerationConfig(**config_dict)
    
    # Load from environment variables
    config_dict = {
        "total_cases": int(os.getenv("TOTAL_CASES", 1200)),
        "batch_size": int(os.getenv("BATCH_SIZE", 10)),
        "model_id": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
        "region": os.getenv("BEDROCK_REGION", "us-east-1"),
        "max_retries": int(os.getenv("MAX_RETRIES", 3))
    }
    
    return GenerationConfig(**config_dict)


def main():
    """Main function to generate synthetic maltreatment data"""
    
    parser = argparse.ArgumentParser(description="Generate synthetic maltreatment case data")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output", type=str, default="./output/synthetic_maltreatment_data.csv", 
                       help="Output file path")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--cases", type=int, help="Number of cases to generate (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override with command line arguments if provided
        if args.cases:
            config.total_cases = args.cases
        if args.batch_size:
            config.batch_size = args.batch_size
        
        logger.info(f"Starting synthetic data generation with config: {config}")
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize generator
        generator = SyntheticDataGenerator(config)
        
        # Generate dataset
        df = generator.generate_dataset(args.output)
        
        # Get and display statistics
        stats = generator.get_generation_stats()
        logger.info("Generation Statistics:")
        logger.info(json.dumps(stats, indent=2))
        
        # Save statistics
        stats_path = output_path.parent / f"generation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset generation completed successfully!")
        logger.info(f"Dataset saved to: {args.output}")
        logger.info(f"Statistics saved to: {stats_path}")
        logger.info(f"Total cases generated: {len(df)}")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
