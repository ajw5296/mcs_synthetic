import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
from pathlib import Path


class DatasetAnalyzer:
    """Analyze generated synthetic dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def generate_summary_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive summary report of the dataset"""
        
        report = {
            "dataset_overview": self._get_dataset_overview(),
            "binary_coding_analysis": self._analyze_binary_codings(),
            "severity_analysis": self._analyze_severity_codings(),
            "narrative_analysis": self._analyze_narratives(),
            "data_quality_checks": self._perform_quality_checks(),
            "correlations": self._analyze_correlations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _get_dataset_overview(self) -> Dict[str, Any]:
        """Get basic dataset overview"""
        
        return {
            "total_cases": len(self.df),
            "total_columns": len(self.df.columns),
            "narrative_columns": 4,  # maltreatment, severity, risk, safety
            "binary_coding_columns": 8,  # all binary abuse type indicators
            "severity_coding_columns": 7,  # severity ratings for each abuse type
            "missing_values": self.df.isnull().sum().to_dict()
        }
    
    def _analyze_binary_codings(self) -> Dict[str, Any]:
        """Analyze binary coding distributions"""
        
        binary_columns = [
            'any_maltreatment', 'sexual_abuse', 'physical_abuse',
            'physical_neglect_failure_to_provide', 'physical_neglect_lack_of_supervision',
            'emotional_abuse', 'moral_legal_abuse', 'educational_abuse'
        ]
        
        analysis = {}
        
        for col in binary_columns:
            if col in self.df.columns:
                counts = self.df[col].value_counts()
                total = len(self.df)
                
                analysis[col] = {
                    "true_count": int(counts.get(True, 0)),
                    "false_count": int(counts.get(False, 0)),
                    "true_percentage": float(counts.get(True, 0) / total * 100),
                    "false_percentage": float(counts.get(False, 0) / total * 100)
                }
        
        return analysis
    
    def _analyze_severity_codings(self) -> Dict[str, Any]:
        """Analyze severity coding distributions"""
        
        severity_columns = [
            'sexual_abuse_severity', 'physical_abuse_severity',
            'physical_neglect_failure_severity', 'physical_neglect_supervision_severity',
            'emotional_abuse_severity', 'moral_legal_abuse_severity',
            'educational_abuse_severity'
        ]
        
        analysis = {}
        
        for col in severity_columns:
            if col in self.df.columns:
                # Only analyze non-null values
                non_null_data = self.df[col].dropna()
                
                if len(non_null_data) > 0:
                    analysis[col] = {
                        "count": int(len(non_null_data)),
                        "mean": float(non_null_data.mean()),
                        "std": float(non_null_data.std()),
                        "min": int(non_null_data.min()),
                        "max": int(non_null_data.max()),
                        "distribution": {
                            str(i): int(sum(non_null_data == i)) for i in range(1, 6)
                        }
                    }
                else:
                    analysis[col] = {"count": 0, "note": "No cases with this severity coding"}
        
        return analysis
    
    def _analyze_narratives(self) -> Dict[str, Any]:
        """Analyze narrative text characteristics"""
        
        narrative_columns = [
            'maltreatment_narrative', 'severity_narrative',
            'risk_narrative', 'safety_assessment_narrative'
        ]
        
        analysis = {}
        
        for col in narrative_columns:
            if col in self.df.columns:
                texts = self.df[col].dropna()
                
                if len(texts) > 0:
                    word_counts = texts.apply(lambda x: len(str(x).split()))
                    char_counts = texts.apply(lambda x: len(str(x)))
                    
                    analysis[col] = {
                        "count": int(len(texts)),
                        "avg_word_count": float(word_counts.mean()),
                        "avg_char_count": float(char_counts.mean()),
                        "min_word_count": int(word_counts.min()),
                        "max_word_count": int(word_counts.max()),
                        "word_count_std": float(word_counts.std())
                    }
        
        return analysis
    
    def _perform_quality_checks(self) -> Dict[str, Any]:
        """Perform data quality checks"""
        
        checks = {
            "consistency_checks": {},
            "validation_errors": [],
            "completeness": {}
        }
        
        # Check consistency between binary and severity codings
        binary_severity_pairs = [
            ('sexual_abuse', 'sexual_abuse_severity'),
            ('physical_abuse', 'physical_abuse_severity'),
            ('physical_neglect_failure_to_provide', 'physical_neglect_failure_severity'),
            ('physical_neglect_lack_of_supervision', 'physical_neglect_supervision_severity'),
            ('emotional_abuse', 'emotional_abuse_severity'),
            ('moral_legal_abuse', 'moral_legal_abuse_severity'),
            ('educational_abuse', 'educational_abuse_severity')
        ]
        
        for binary_col, severity_col in binary_severity_pairs:
            if binary_col in self.df.columns and severity_col in self.df.columns:
                # Cases where binary is True but severity is null
                inconsistent_1 = sum((self.df[binary_col] == True) & (self.df[severity_col].isnull()))
                
                # Cases where binary is False but severity is not null
                inconsistent_2 = sum((self.df[binary_col] == False) & (self.df[severity_col].notnull()))
                
                checks["consistency_checks"][f"{binary_col}_consistency"] = {
                    "binary_true_severity_null": int(inconsistent_1),
                    "binary_false_severity_not_null": int(inconsistent_2),
                    "total_inconsistent": int(inconsistent_1 + inconsistent_2)
                }
        
        # Check that any_maltreatment is consistent
        if 'any_maltreatment' in self.df.columns:
            other_abuse_cols = [
                'sexual_abuse', 'physical_abuse', 'physical_neglect_failure_to_provide',
                'physical_neglect_lack_of_supervision', 'emotional_abuse', 
                'moral_legal_abuse', 'educational_abuse'
            ]
            
            # Cases where any_maltreatment is False but other abuse types are True
            for col in other_abuse_cols:
                if col in self.df.columns:
                    inconsistent = sum((self.df['any_maltreatment'] == False) & (self.df[col] == True))
                    if inconsistent > 0:
                        checks["validation_errors"].append(
                            f"{inconsistent} cases where any_maltreatment is False but {col} is True"
                        )
        
        # Check completeness
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            checks["completeness"][col] = {
                "non_null_count": int(len(self.df) - null_count),
                "null_count": int(null_count),
                "completeness_percentage": float((len(self.df) - null_count) / len(self.df) * 100)
            }
        
        return checks
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different variables"""
        
        # Get numeric columns only
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 1:
            correlation_matrix = self.df[numeric_columns].corr()
            
            # Convert to dictionary format
            correlations = {}
            for col1 in correlation_matrix.columns:
                correlations[col1] = {}
                for col2 in correlation_matrix.columns:
                    if col1 != col2:
                        correlations[col1][col2] = float(correlation_matrix.loc[col1, col2])
            
            return correlations
        
        return {}
    
    def create_visualizations(self, output_dir: str = "./visualizations"):
        """Create visualization plots for the dataset"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Binary coding distribution
        self._plot_binary_distributions(output_path)
        
        # 2. Severity distributions
        self._plot_severity_distributions(output_path)
        
        # 3. Narrative length distributions
        self._plot_narrative_lengths(output_path)
        
        # 4. Co-occurrence matrix
        self._plot_cooccurrence_matrix(output_path)
        
        print(f"Visualizations saved to: {output_path}")
    
    def _plot_binary_distributions(self, output_path: Path):
        """Plot binary coding distributions"""
        
        binary_columns = [
            'any_maltreatment', 'sexual_abuse', 'physical_abuse',
            'physical_neglect_failure_to_provide', 'physical_neglect_lack_of_supervision',
            'emotional_abuse', 'moral_legal_abuse', 'educational_abuse'
        ]
        
        available_cols = [col for col in binary_columns if col in self.df.columns]
        
        if available_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            true_counts = [self.df[col].sum() for col in available_cols]
            
            bars = ax.bar(range(len(available_cols)), true_counts)
            ax.set_xlabel('Abuse Types')
            ax.set_ylabel('Number of True Cases')
            ax.set_title('Distribution of Abuse Types (Binary Codings)')
            ax.set_xticks(range(len(available_cols)))
            ax.set_xticklabels([col.replace('_', '\n') for col in available_cols], rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, true_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(count)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'binary_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_severity_distributions(self, output_path: Path):
        """Plot severity distributions"""
        
        severity_columns = [
            'sexual_abuse_severity', 'physical_abuse_severity',
            'physical_neglect_failure_severity', 'physical_neglect_supervision_severity',
            'emotional_abuse_severity', 'moral_legal_abuse_severity',
            'educational_abuse_severity'
        ]
        
        available_cols = [col for col in severity_columns if col in self.df.columns]
        
        if available_cols:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            for i, col in enumerate(available_cols[:7]):  # Max 7 plots
                if i < len(axes):
                    data = self.df[col].dropna()
                    if len(data) > 0:
                        axes[i].hist(data, bins=range(1, 7), alpha=0.7, edgecolor='black')
                        axes[i].set_xlabel('Severity Level')
                        axes[i].set_ylabel('Frequency')
                        axes[i].set_title(col.replace('_', ' ').title())
                        axes[i].set_xticks(range(1, 6))
            
            # Hide empty subplots
            for i in range(len(available_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path / 'severity_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_narrative_lengths(self, output_path: Path):
        """Plot narrative length distributions"""
        
        narrative_columns = [
            'maltreatment_narrative', 'severity_narrative',
            'risk_narrative', 'safety_assessment_narrative'
        ]
        
        available_cols = [col for col in narrative_columns if col in self.df.columns]
        
        if available_cols:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, col in enumerate(available_cols[:4]):
                word_counts = self.df[col].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
                
                axes[i].hist(word_counts, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel('Word Count')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{col.replace("_", " ").title()} - Word Count Distribution')
                axes[i].axvline(word_counts.mean(), color='red', linestyle='--', 
                               label=f'Mean: {word_counts.mean():.1f}')
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig(output_path / 'narrative_lengths.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_cooccurrence_matrix(self, output_path: Path):
        """Plot co-occurrence matrix of abuse types"""
        
        binary_columns = [
            'sexual_abuse', 'physical_abuse',
            'physical_neglect_failure_to_provide', 'physical_neglect_lack_of_supervision',
            'emotional_abuse', 'moral_legal_abuse', 'educational_abuse'
        ]
        
        available_cols = [col for col in binary_columns if col in self.df.columns]
        
        if len(available_cols) > 1:
            # Create co-occurrence matrix
            cooccurrence = np.zeros((len(available_cols), len(available_cols)))
            
            for i, col1 in enumerate(available_cols):
                for j, col2 in enumerate(available_cols):
                    cooccurrence[i, j] = sum((self.df[col1] == True) & (self.df[col2] == True))
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(cooccurrence, 
                       xticklabels=[col.replace('_', '\n') for col in available_cols],
                       yticklabels=[col.replace('_', '\n') for col in available_cols],
                       annot=True, fmt='g', cmap='Blues', ax=ax)
            
            ax.set_title('Co-occurrence Matrix of Abuse Types')
            plt.tight_layout()
            plt.savefig(output_path / 'cooccurrence_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
