from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class AbuseType(str, Enum):
    SEXUAL = "sexual_abuse"
    PHYSICAL = "physical_abuse"
    PHYSICAL_NEGLECT_FAILURE = "physical_neglect_failure_to_provide"
    PHYSICAL_NEGLECT_SUPERVISION = "physical_neglect_lack_of_supervision"
    EMOTIONAL = "emotional_abuse"
    MORAL_LEGAL = "moral_legal_abuse"
    EDUCATIONAL = "educational_abuse"


class BinaryCodings(BaseModel):
    """Binary yes/no codings for each abuse type"""
    any_maltreatment: bool = Field(description="Did any maltreatment happen?")
    sexual_abuse: bool = Field(description="Was it sexual abuse?")
    physical_abuse: bool = Field(description="Was it physical abuse?")
    physical_neglect_failure_to_provide: bool = Field(description="Was it physical neglect: failure to provide?")
    physical_neglect_lack_of_supervision: bool = Field(description="Was it physical neglect: lack of supervision?")
    emotional_abuse: bool = Field(description="Was it emotional abuse?")
    moral_legal_abuse: bool = Field(description="Was it moral or legal abuse?")
    educational_abuse: bool = Field(description="Was it educational abuse?")


class SeverityCodings(BaseModel):
    """Severity ratings on 1-5 scale for each abuse type (when applicable)"""
    sexual_abuse_severity: Optional[int] = Field(None, ge=1, le=5, description="Sexual abuse severity (1-5)")
    physical_abuse_severity: Optional[int] = Field(None, ge=1, le=5, description="Physical abuse severity (1-5)")
    physical_neglect_failure_severity: Optional[int] = Field(None, ge=1, le=5, description="Physical neglect: failure to provide severity (1-5)")
    physical_neglect_supervision_severity: Optional[int] = Field(None, ge=1, le=5, description="Physical neglect: lack of supervision severity (1-5)")
    emotional_abuse_severity: Optional[int] = Field(None, ge=1, le=5, description="Emotional abuse severity (1-5)")
    moral_legal_abuse_severity: Optional[int] = Field(None, ge=1, le=5, description="Moral/legal abuse severity (1-5)")
    educational_abuse_severity: Optional[int] = Field(None, ge=1, le=5, description="Educational abuse severity (1-5)")


class NarrativeData(BaseModel):
    """Container for all narrative text data"""
    maltreatment_narrative: str = Field(description="Detailed case worker account of what transpired")
    severity_narrative: str = Field(description="Case worker's written assessment of case severity")
    risk_narrative: str = Field(description="Case worker's assessment of child's risk of future abuse")
    safety_assessment_narrative: str = Field(description="Case worker's assessment of child's current safety")


class MaltreatmentCase(BaseModel):
    """Complete maltreatment case with narratives and codings"""
    case_id: str = Field(description="Unique case identifier")
    narratives: NarrativeData
    binary_codings: BinaryCodings
    severity_codings: SeverityCodings
    
    def validate_codings(self) -> bool:
        """Validate that severity codings are only present when binary codings are True"""
        valid = True
        
        if not self.binary_codings.sexual_abuse and self.severity_codings.sexual_abuse_severity is not None:
            valid = False
        if not self.binary_codings.physical_abuse and self.severity_codings.physical_abuse_severity is not None:
            valid = False
        if not self.binary_codings.physical_neglect_failure_to_provide and self.severity_codings.physical_neglect_failure_severity is not None:
            valid = False
        if not self.binary_codings.physical_neglect_lack_of_supervision and self.severity_codings.physical_neglect_supervision_severity is not None:
            valid = False
        if not self.binary_codings.emotional_abuse and self.severity_codings.emotional_abuse_severity is not None:
            valid = False
        if not self.binary_codings.moral_legal_abuse and self.severity_codings.moral_legal_abuse_severity is not None:
            valid = False
        if not self.binary_codings.educational_abuse and self.severity_codings.educational_abuse_severity is not None:
            valid = False
            
        return valid


class GenerationConfig(BaseModel):
    """Configuration for data generation"""
    total_cases: int = Field(default=1200, description="Total number of cases to generate")
    batch_size: int = Field(default=10, description="Number of cases to generate in each batch")
    model_id: str = Field(default="anthropic.claude-3-sonnet-20240229-v1:0", description="AWS Bedrock model ID")
    region: str = Field(default="us-east-1", description="AWS region")
    max_retries: int = Field(default=3, description="Maximum retries for failed generations")
    
    # Distribution settings for abuse types (percentages)
    abuse_type_distribution: Dict[str, float] = Field(default={
        "no_maltreatment": 0.15,  # 15% no maltreatment cases
        "sexual_abuse": 0.12,     # 12% sexual abuse cases
        "physical_abuse": 0.20,   # 20% physical abuse cases  
        "physical_neglect_failure": 0.15,  # 15% failure to provide cases
        "physical_neglect_supervision": 0.13,  # 13% lack of supervision cases
        "emotional_abuse": 0.10,  # 10% emotional abuse cases
        "moral_legal_abuse": 0.08,  # 8% moral/legal abuse cases
        "educational_abuse": 0.07,  # 7% educational abuse cases
    })
    
    # Severity distribution for each abuse type
    severity_distribution: Dict[int, float] = Field(default={
        1: 0.05,  # 5% severity 1 (very mild)
        2: 0.15,  # 15% severity 2 (mild)
        3: 0.35,  # 35% severity 3 (moderate)
        4: 0.30,  # 30% severity 4 (severe)
        5: 0.15,  # 15% severity 5 (very severe)
    })
