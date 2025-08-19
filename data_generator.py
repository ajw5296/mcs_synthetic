import pandas as pd
import numpy as np
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import random
import time
from datetime import datetime

from models import (
    MaltreatmentCase, NarrativeData, BinaryCodings, 
    SeverityCodings, GenerationConfig, AbuseType
)
from bedrock_client import BedrockClient


class SyntheticDataGenerator:
    """Main class for generating synthetic maltreatment case data"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.bedrock_client = BedrockClient(
            region=config.region,
            model_id=config.model_id
        )
        self.logger = logging.getLogger(__name__)
        self.generated_cases: List[MaltreatmentCase] = []
        
    def generate_dataset(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        
        self.logger.info(f"Starting generation of {self.config.total_cases} cases")
        
        # Calculate how many cases of each type to generate
        case_distribution = self._calculate_case_distribution()
        
        # Generate cases in batches
        for abuse_type, count in case_distribution.items():
            if count > 0:
                self.logger.info(f"Generating {count} cases of type: {abuse_type}")
                self._generate_cases_for_type(abuse_type, count)
        
        # Convert to DataFrame
        df = self._convert_to_dataframe()
        
        # Always save to CSV with default name if no output path provided
        if not output_path:
            output_path = "./output/datasets/synthetic_maltreatment_data.csv"
        
        # Save to file
        self._save_dataset(df, output_path)
            
        self.logger.info(f"Generated {len(self.generated_cases)} total cases")
        return df
    
    def _calculate_case_distribution(self) -> Dict[str, int]:
        """Calculate how many cases to generate for each abuse type"""
        
        distribution = {}
        remaining_cases = self.config.total_cases
        
        # Calculate exact counts for each type
        for abuse_type, percentage in self.config.abuse_type_distribution.items():
            count = int(self.config.total_cases * percentage)
            distribution[abuse_type] = count
            remaining_cases -= count
        
        # Distribute any remaining cases randomly
        abuse_types = list(self.config.abuse_type_distribution.keys())
        while remaining_cases > 0:
            abuse_type = random.choice(abuse_types)
            distribution[abuse_type] += 1
            remaining_cases -= 1
        
        return distribution
    
    def _generate_cases_for_type(self, abuse_type: str, count: int):
        """Generate cases for a specific abuse type"""
        
        batch_size = min(self.config.batch_size, count)
        batches = [batch_size] * (count // batch_size)
        if count % batch_size > 0:
            batches.append(count % batch_size)
        
        for batch_count in tqdm(batches, desc=f"Generating {abuse_type} cases"):
            batch_cases = []
            
            for _ in range(batch_count):
                try:
                    case = self._generate_single_case(abuse_type)
                    if case and case.validate_codings():
                        batch_cases.append(case)
                    else:
                        self.logger.warning(f"Generated invalid case for {abuse_type}, skipping")
                        
                except Exception as e:
                    self.logger.error(f"Error generating case for {abuse_type}: {str(e)}")
                    continue
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            self.generated_cases.extend(batch_cases)
            self.logger.info(f"Completed batch: {len(batch_cases)} valid cases generated")
    
    def _generate_single_case(self, abuse_type: str) -> Optional[MaltreatmentCase]:
        """Generate a single maltreatment case"""
        
        try:
            # Generate severity if applicable
            severity = None
            if abuse_type != "no_maltreatment":
                severity = self._sample_severity()
            
            # Generate narratives
            narratives_dict = self.bedrock_client.generate_case_narratives(
                abuse_type=abuse_type,
                severity=severity,
                max_retries=self.config.max_retries
            )
            
            # Generate codings based on narratives
            codings_dict = self.bedrock_client.generate_codings(
                narratives=narratives_dict,
                max_retries=self.config.max_retries
            )
            
            # Create case object
            case = MaltreatmentCase(
                case_id=str(uuid.uuid4()),
                narratives=NarrativeData(**narratives_dict),
                binary_codings=BinaryCodings(**codings_dict['binary_codings']),
                severity_codings=SeverityCodings(**codings_dict['severity_codings'])
            )
            
            return case
            
        except Exception as e:
            self.logger.error(f"Error generating case: {str(e)}")
            return None
    
    def _sample_severity(self) -> int:
        """Sample a severity level based on distribution"""
        severities = list(self.config.severity_distribution.keys())
        weights = list(self.config.severity_distribution.values())
        return np.random.choice(severities, p=weights)
    
    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Convert generated cases to pandas DataFrame"""
        
        data = []
        
        for case in self.generated_cases:
            row = {
                'case_id': case.case_id,
                
                # Narratives
                'maltreatment_narrative': case.narratives.maltreatment_narrative,
                'severity_narrative': case.narratives.severity_narrative,
                'risk_narrative': case.narratives.risk_narrative,
                'safety_assessment_narrative': case.narratives.safety_assessment_narrative,
                
                # Binary codings
                'any_maltreatment': case.binary_codings.any_maltreatment,
                'sexual_abuse': case.binary_codings.sexual_abuse,
                'physical_abuse': case.binary_codings.physical_abuse,
                'physical_neglect_failure_to_provide': case.binary_codings.physical_neglect_failure_to_provide,
                'physical_neglect_lack_of_supervision': case.binary_codings.physical_neglect_lack_of_supervision,
                'emotional_abuse': case.binary_codings.emotional_abuse,
                'moral_legal_abuse': case.binary_codings.moral_legal_abuse,
                'educational_abuse': case.binary_codings.educational_abuse,
                
                # Severity codings
                'sexual_abuse_severity': case.severity_codings.sexual_abuse_severity,
                'physical_abuse_severity': case.severity_codings.physical_abuse_severity,
                'physical_neglect_failure_severity': case.severity_codings.physical_neglect_failure_severity,
                'physical_neglect_supervision_severity': case.severity_codings.physical_neglect_supervision_severity,
                'emotional_abuse_severity': case.severity_codings.emotional_abuse_severity,
                'moral_legal_abuse_severity': case.severity_codings.moral_legal_abuse_severity,
                'educational_abuse_severity': case.severity_codings.educational_abuse_severity,
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _save_dataset(self, df: pd.DataFrame, output_path: str):
        """Save dataset to file with timestamp"""
        
        output_path_obj = Path(output_path)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path_obj.stem
        suffix = output_path_obj.suffix or '.csv'
        
        # Create timestamped filename: original_name_YYYYMMDD_HHMMSS.ext
        timestamped_name = f"{stem}_{timestamp}{suffix}"
        final_output_path = output_path_obj.parent / timestamped_name
        
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if final_output_path.suffix == '.csv':
            df.to_csv(final_output_path, index=False)
        elif final_output_path.suffix in ['.xlsx', '.xls']:
            df.to_excel(final_output_path, index=False)
        elif final_output_path.suffix == '.json':
            df.to_json(final_output_path, orient='records', indent=2)
        elif final_output_path.suffix == '.parquet':
            df.to_parquet(final_output_path, index=False)
        else:
            # Default to CSV
            df.to_csv(final_output_path.with_suffix('.csv'), index=False)
        
        self.logger.info(f"Dataset saved to: {final_output_path}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generated dataset"""
        
        if not self.generated_cases:
            return {}
        
        stats = {
            'total_cases': len(self.generated_cases),
            'binary_coding_counts': {},
            'severity_distributions': {},
            'validation_results': {
                'valid_cases': 0,
                'invalid_cases': 0
            }
        }
        
        # Count binary codings
        binary_fields = [
            'any_maltreatment', 'sexual_abuse', 'physical_abuse',
            'physical_neglect_failure_to_provide', 'physical_neglect_lack_of_supervision',
            'emotional_abuse', 'moral_legal_abuse', 'educational_abuse'
        ]
        
        for field in binary_fields:
            stats['binary_coding_counts'][field] = sum(
                1 for case in self.generated_cases 
                if getattr(case.binary_codings, field)
            )
        
        # Severity distributions
        severity_fields = [
            'sexual_abuse_severity', 'physical_abuse_severity',
            'physical_neglect_failure_severity', 'physical_neglect_supervision_severity',
            'emotional_abuse_severity', 'moral_legal_abuse_severity',
            'educational_abuse_severity'
        ]
        
        for field in severity_fields:
            severities = [
                getattr(case.severity_codings, field)
                for case in self.generated_cases
                if getattr(case.severity_codings, field) is not None
            ]
            
            if severities:
                stats['severity_distributions'][field] = {
                    'count': len(severities),
                    'mean': np.mean(severities),
                    'distribution': {
                        str(i): severities.count(i) for i in range(1, 6)
                    }
                }
        
        # Validation results
        for case in self.generated_cases:
            if case.validate_codings():
                stats['validation_results']['valid_cases'] += 1
            else:
                stats['validation_results']['invalid_cases'] += 1
        
        return stats
