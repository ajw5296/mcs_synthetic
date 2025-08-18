import boto3
import json
import logging
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
import time
import random


class BedrockClient:
    """AWS Bedrock client for generating synthetic data"""
    
    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        self.region = region
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.logger = logging.getLogger(__name__)
        
    def generate_case_narratives(self, abuse_type: str, severity: Optional[int] = None, 
                               max_retries: int = 3) -> Dict[str, str]:
        """Generate all four narrative types for a maltreatment case"""
        
        prompt = self._build_narrative_prompt(abuse_type, severity)
        
        for attempt in range(max_retries):
            try:
                response = self._call_bedrock(prompt)
                narratives = self._parse_narrative_response(response)
                
                if self._validate_narratives(narratives):
                    return narratives
                else:
                    self.logger.warning(f"Invalid narratives generated, attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.error(f"Error generating narratives, attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 3))  # Random backoff
                    
        raise Exception(f"Failed to generate valid narratives after {max_retries} attempts")
    
    def generate_codings(self, narratives: Dict[str, str], max_retries: int = 3) -> Dict[str, Any]:
        """Generate binary and severity codings based on narratives"""
        
        prompt = self._build_coding_prompt(narratives)
        
        for attempt in range(max_retries):
            try:
                response = self._call_bedrock(prompt)
                codings = self._parse_coding_response(response)
                
                if self._validate_codings(codings):
                    return codings
                else:
                    self.logger.warning(f"Invalid codings generated, attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.error(f"Error generating codings, attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 3))
                    
        raise Exception(f"Failed to generate valid codings after {max_retries} attempts")
    
    def _build_narrative_prompt(self, abuse_type: str, severity: Optional[int] = None) -> str:
        """Build prompt for generating narratives"""
        
        severity_desc = ""
        if severity and severity > 0:
            severity_levels = {
                1: "very mild",
                2: "mild", 
                3: "moderate",
                4: "severe",
                5: "very severe"
            }
            severity_desc = f" The case should be {severity_levels.get(severity, 'moderate')} in severity."
        
        abuse_descriptions = {
            "no_maltreatment": "a case where allegations were made but no actual maltreatment occurred",
            "sexual_abuse": "a case involving sexual abuse or exploitation of a child",
            "physical_abuse": "a case involving physical abuse, hitting, beating, or other physical harm",
            "physical_neglect_failure_to_provide": "a case involving failure to provide basic needs like food, shelter, clothing, or medical care",
            "physical_neglect_lack_of_supervision": "a case involving lack of proper supervision resulting in risk or harm to a child",
            "emotional_abuse": "a case involving emotional, psychological, or verbal abuse",
            "moral_legal_abuse": "a case involving exposure to illegal activities, drugs, or immoral behavior",
            "educational_abuse": "a case involving educational neglect or preventing a child from attending school"
        }
        
        abuse_desc = abuse_descriptions.get(abuse_type, "a general maltreatment case")
        
        prompt = f"""You are a social worker creating realistic synthetic case narratives for research purposes. Generate narratives for {abuse_desc}.{severity_desc}

Please create four distinct narrative sections:

1. MALTREATMENT_NARRATIVE: A detailed account (200-400 words) written by the case worker describing what transpired. Include allegations from various sources like neighbors, police, parents, siblings, friends, etc. This should be a comprehensive accounting of all available information about the incident(s).

2. SEVERITY_NARRATIVE: A more concise assessment (100-200 words) by the case worker evaluating how severe the case is. Focus on the key factors that determine severity.

3. RISK_NARRATIVE: An assessment (100-200 words) of whether the child is at risk of future abuse. Consider protective factors and risk factors.

4. SAFETY_ASSESSMENT_NARRATIVE: An evaluation (100-200 words) of whether the child is currently safe. Focus on immediate safety concerns and protective measures.

Make the narratives realistic and consistent with each other. Use professional social work language but make it authentic. Vary the writing style slightly between narratives as different sections might be written at different times.

Format your response as JSON with exactly these keys:
{{
    "maltreatment_narrative": "...",
    "severity_narrative": "...", 
    "risk_narrative": "...",
    "safety_assessment_narrative": "..."
}}"""

        return prompt
    
    def _build_coding_prompt(self, narratives: Dict[str, str]) -> str:
        """Build prompt for generating codings based on narratives"""
        
        prompt = f"""You are an expert social worker who codes maltreatment cases based on narrative data. Based on the following narratives, provide accurate binary codings and severity ratings.

NARRATIVES:
Maltreatment Narrative: {narratives['maltreatment_narrative']}

Severity Narrative: {narratives['severity_narrative']}

Risk Narrative: {narratives['risk_narrative']}

Safety Assessment Narrative: {narratives['safety_assessment_narrative']}

Please provide codings in the following JSON format:

{{
    "binary_codings": {{
        "any_maltreatment": true/false,
        "sexual_abuse": true/false,
        "physical_abuse": true/false,
        "physical_neglect_failure_to_provide": true/false,
        "physical_neglect_lack_of_supervision": true/false,
        "emotional_abuse": true/false,
        "moral_legal_abuse": true/false,
        "educational_abuse": true/false
    }},
    "severity_codings": {{
        "sexual_abuse_severity": 1-5 or null,
        "physical_abuse_severity": 1-5 or null,
        "physical_neglect_failure_severity": 1-5 or null,
        "physical_neglect_supervision_severity": 1-5 or null,
        "emotional_abuse_severity": 1-5 or null,
        "moral_legal_abuse_severity": 1-5 or null,
        "educational_abuse_severity": 1-5 or null
    }}
}}

IMPORTANT RULES:
1. Only set severity ratings (1-5) for abuse types that are coded as true in binary_codings
2. Set severity ratings to null for abuse types coded as false
3. If any_maltreatment is false, all other binary codings should be false and all severity ratings should be null
4. Severity scale: 1=very mild, 2=mild, 3=moderate, 4=severe, 5=very severe
5. Base your codings strictly on what is described in the narratives"""

        return prompt
    
    def _call_bedrock(self, prompt: str) -> str:
        """Make API call to Bedrock"""
        
        if "anthropic.claude" in self.model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.9
            }
        else:
            # Fallback for other models
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 2000,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            if "anthropic.claude" in self.model_id:
                return response_body['content'][0]['text']
            else:
                return response_body.get('results', [{}])[0].get('outputText', '')
                
        except ClientError as e:
            self.logger.error(f"AWS Bedrock API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error calling Bedrock: {str(e)}")
            raise
    
    def _parse_narrative_response(self, response: str) -> Dict[str, str]:
        """Parse narrative response from Bedrock"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse narrative JSON: {str(e)}")
            raise
    
    def _parse_coding_response(self, response: str) -> Dict[str, Any]:
        """Parse coding response from Bedrock"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse coding JSON: {str(e)}")
            raise
    
    def _validate_narratives(self, narratives: Dict[str, str]) -> bool:
        """Validate that all required narrative fields are present and non-empty"""
        required_fields = [
            'maltreatment_narrative',
            'severity_narrative', 
            'risk_narrative',
            'safety_assessment_narrative'
        ]
        
        for field in required_fields:
            if field not in narratives or not narratives[field] or len(narratives[field].strip()) < 50:
                return False
                
        return True
    
    def _validate_codings(self, codings: Dict[str, Any]) -> bool:
        """Validate codings structure and logic"""
        try:
            binary = codings.get('binary_codings', {})
            severity = codings.get('severity_codings', {})
            
            # Check that binary codings are boolean
            binary_fields = [
                'any_maltreatment', 'sexual_abuse', 'physical_abuse',
                'physical_neglect_failure_to_provide', 'physical_neglect_lack_of_supervision',
                'emotional_abuse', 'moral_legal_abuse', 'educational_abuse'
            ]
            
            for field in binary_fields:
                if field not in binary or not isinstance(binary[field], bool):
                    return False
            
            # Check severity codings logic
            severity_mappings = {
                'sexual_abuse': 'sexual_abuse_severity',
                'physical_abuse': 'physical_abuse_severity',
                'physical_neglect_failure_to_provide': 'physical_neglect_failure_severity',
                'physical_neglect_lack_of_supervision': 'physical_neglect_supervision_severity',
                'emotional_abuse': 'emotional_abuse_severity',
                'moral_legal_abuse': 'moral_legal_abuse_severity',
                'educational_abuse': 'educational_abuse_severity'
            }
            
            for binary_field, severity_field in severity_mappings.items():
                if binary[binary_field]:
                    # If binary is True, severity should be 1-5
                    sev_val = severity.get(severity_field)
                    if sev_val is None or not isinstance(sev_val, int) or sev_val < 1 or sev_val > 5:
                        return False
                else:
                    # If binary is False, severity should be null
                    if severity.get(severity_field) is not None:
                        return False
            
            # If no maltreatment, all other fields should be False/null
            if not binary['any_maltreatment']:
                for field in binary_fields[1:]:  # Skip 'any_maltreatment'
                    if binary[field]:
                        return False
                        
                for sev_field in severity_mappings.values():
                    if severity.get(sev_field) is not None:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating codings: {str(e)}")
            return False
