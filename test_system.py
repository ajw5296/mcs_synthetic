#!/usr/bin/env python3
"""
Test script to verify the synthetic data generation system
This can be run to test the system with a small sample before full generation
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import GenerationConfig, MaltreatmentCase, NarrativeData, BinaryCodings, SeverityCodings
from data_generator import SyntheticDataGenerator
from analyzer import DatasetAnalyzer


def test_models():
    """Test data models and validation"""
    print("Testing data models...")
    
    # Test valid case
    case = MaltreatmentCase(
        case_id="test-001",
        narratives=NarrativeData(
            maltreatment_narrative="Test narrative about a case involving physical abuse...",
            severity_narrative="This case shows moderate severity...",
            risk_narrative="The child appears to be at moderate risk...",
            safety_assessment_narrative="Current safety measures are in place..."
        ),
        binary_codings=BinaryCodings(
            any_maltreatment=True,
            sexual_abuse=False,
            physical_abuse=True,
            physical_neglect_failure_to_provide=False,
            physical_neglect_lack_of_supervision=False,
            emotional_abuse=False,
            moral_legal_abuse=False,
            educational_abuse=False
        ),
        severity_codings=SeverityCodings(
            sexual_abuse_severity=None,
            physical_abuse_severity=3,
            physical_neglect_failure_severity=None,
            physical_neglect_supervision_severity=None,
            emotional_abuse_severity=None,
            moral_legal_abuse_severity=None,
            educational_abuse_severity=None
        )
    )
    
    is_valid = case.validate_codings()
    print(f"Valid case validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Test invalid case (binary false but severity present)
    invalid_case = MaltreatmentCase(
        case_id="test-002",
        narratives=NarrativeData(
            maltreatment_narrative="Test narrative...",
            severity_narrative="Test severity...",
            risk_narrative="Test risk...",
            safety_assessment_narrative="Test safety..."
        ),
        binary_codings=BinaryCodings(
            any_maltreatment=True,
            sexual_abuse=False,  # False
            physical_abuse=False,
            physical_neglect_failure_to_provide=False,
            physical_neglect_lack_of_supervision=False,
            emotional_abuse=False,
            moral_legal_abuse=False,
            educational_abuse=False
        ),
        severity_codings=SeverityCodings(
            sexual_abuse_severity=3,  # But severity is present - should be invalid
            physical_abuse_severity=None,
            physical_neglect_failure_severity=None,
            physical_neglect_supervision_severity=None,
            emotional_abuse_severity=None,
            moral_legal_abuse_severity=None,
            educational_abuse_severity=None
        )
    )
    
    is_invalid = not invalid_case.validate_codings()  # Should be invalid
    print(f"Invalid case detection: {'✓ PASS' if is_invalid else '✗ FAIL'}")


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    config = GenerationConfig()
    print(f"Default config loaded: ✓ PASS")
    print(f"  Total cases: {config.total_cases}")
    print(f"  Model ID: {config.model_id}")
    print(f"  Region: {config.region}")
    
    # Test distribution sums to approximately 1.0
    total_distribution = sum(config.abuse_type_distribution.values())
    is_valid_dist = 0.99 <= total_distribution <= 1.01
    print(f"Valid distribution (sum={total_distribution:.3f}): {'✓ PASS' if is_valid_dist else '✗ FAIL'}")


def test_aws_connection():
    """Test AWS Bedrock connection (without making actual calls)"""
    print("\nTesting AWS connection setup...")
    
    try:
        from bedrock_client import BedrockClient
        
        # Create client (this will test import and basic setup)
        client = BedrockClient()
        print("BedrockClient initialization: ✓ PASS")
        
        # Check if AWS credentials are available
        import boto3
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                print("AWS credentials found: ✓ PASS")
            else:
                print("AWS credentials: ✗ NOT FOUND (set up .env file)")
        except Exception as e:
            print(f"AWS credentials check: ✗ ERROR ({str(e)})")
            
    except ImportError as e:
        print(f"Import error: ✗ FAIL ({str(e)})")
    except Exception as e:
        print(f"AWS setup error: ✗ ERROR ({str(e)})")


def test_sample_generation():
    """Test sample generation workflow (without AWS calls)"""
    print("\nTesting sample generation workflow...")
    
    try:
        config = GenerationConfig(
            total_cases=2,
            batch_size=1
        )
        
        generator = SyntheticDataGenerator(config)
        print("SyntheticDataGenerator initialization: ✓ PASS")
        
        # Test distribution calculation
        distribution = generator._calculate_case_distribution()
        total_cases = sum(distribution.values())
        print(f"Case distribution calculation (total={total_cases}): {'✓ PASS' if total_cases == config.total_cases else '✗ FAIL'}")
        
        print("Distribution breakdown:")
        for abuse_type, count in distribution.items():
            if count > 0:
                print(f"  {abuse_type}: {count}")
        
    except Exception as e:
        print(f"Sample generation test: ✗ ERROR ({str(e)})")


def main():
    """Run all tests"""
    print("=== Synthetic Maltreatment Data Generator - System Test ===\n")
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs for testing
    
    test_models()
    test_configuration()
    test_aws_connection()
    test_sample_generation()
    
    print("\n=== Test Summary ===")
    print("If all tests pass, your system is ready for data generation!")
    print("\nNext steps:")
    print("1. Set up your .env file with AWS credentials")
    print("2. Ensure AWS Bedrock access is configured")
    print("3. Run: python main.py --cases 10 --batch-size 2 (for a small test)")
    print("4. Run: python main.py (for full 1200 case generation)")


if __name__ == "__main__":
    main()
