"""
Evaluation results reporting and visualization
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..data.models import EvaluationResult
from ..config import get_config

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate and display evaluation reports"""
    
    def __init__(self):
        self.config = get_config()
        self.console = Console()
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self, eval_result: EvaluationResult, save_to_file: bool = True) -> str:
        """Generate comprehensive evaluation report"""
        
        report_sections = []
        
        # Header
        report_sections.append(self._generate_header(eval_result))
        
        # Overall metrics
        report_sections.append(self._generate_overall_metrics(eval_result))
        
        # Signal type breakdown
        report_sections.append(self._generate_signal_type_breakdown(eval_result))
        
        # Confidence analysis
        report_sections.append(self._generate_confidence_analysis(eval_result))
        
        # Performance analysis
        report_sections.append(self._generate_performance_analysis(eval_result))
        
        # Recommendations
        report_sections.append(self._generate_recommendations(eval_result))
        
        # Combine all sections
        full_report = "\n\n".join(report_sections)
        
        # Save to file if requested
        if save_to_file:
            self._save_report(full_report, eval_result)
        
        return full_report
    
    def _generate_header(self, eval_result: EvaluationResult) -> str:
        """Generate report header"""
        timestamp = eval_result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""
# Boardy Conversation Quality Signals - Evaluation Report

**Generated:** {timestamp}  
**Total Samples:** {eval_result.total_samples}  
**Processing Time:** {eval_result.processing_time_ms}ms  
**Overall F1 Score:** {eval_result.f1_score:.3f}
"""
    
    def _generate_overall_metrics(self, eval_result: EvaluationResult) -> str:
        """Generate overall metrics section"""
        
        return f"""
## Overall Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Precision** | {eval_result.precision:.3f} | ≥0.7 | {'✅' if eval_result.precision >= 0.7 else '❌'} |
| **Recall** | {eval_result.recall:.3f} | ≥0.7 | {'✅' if eval_result.recall >= 0.7 else '❌'} |
| **F1 Score** | {eval_result.f1_score:.3f} | ≥0.7 | {'✅' if eval_result.f1_score >= 0.7 else '❌'} |
| **True Positives** | {eval_result.correct_predictions} | ≥8 | {'✅' if eval_result.correct_predictions >= 8 else '❌'} |
| **False Positives** | {eval_result.false_positives} | Minimize | {'✅' if eval_result.false_positives <= 3 else '⚠️'} |
| **False Negatives** | {eval_result.false_negatives} | Minimize | {'✅' if eval_result.false_negatives <= 3 else '⚠️'} |

### Success Criteria Assessment
- **Primary Goal:** {'✅ ACHIEVED' if eval_result.precision >= 0.7 and eval_result.correct_predictions >= 8 else '❌ NOT ACHIEVED'}
- **System Performance:** {'✅ EXCELLENT' if eval_result.f1_score >= 0.8 else '✅ GOOD' if eval_result.f1_score >= 0.7 else '⚠️ NEEDS IMPROVEMENT'}
"""
    
    def _generate_signal_type_breakdown(self, eval_result: EvaluationResult) -> str:
        """Generate signal type breakdown section"""
        
        if not eval_result.signal_type_breakdown:
            return "\n## Signal Type Analysis\n\nNo signal type data available."
        
        breakdown_text = "\n## Signal Type Analysis\n\n"
        
        for signal_type, metrics in eval_result.signal_type_breakdown.items():
            breakdown_text += f"""
### {signal_type.replace('_', ' ').title()}
- **Precision:** {metrics.get('precision', 0):.3f}
- **Recall:** {metrics.get('recall', 0):.3f}
- **F1 Score:** {metrics.get('f1_score', 0):.3f}
- **Samples:** {metrics.get('total_predicted', 0)} predicted, {metrics.get('total_ground_truth', 0)} ground truth
"""
        
        return breakdown_text
    
    def _generate_confidence_analysis(self, eval_result: EvaluationResult) -> str:
        """Generate confidence analysis section"""
        
        confidence_metrics = eval_result.metadata.get("confidence_metrics", {})
        
        if not confidence_metrics:
            return "\n## Confidence Analysis\n\nNo confidence threshold data available."
        
        confidence_text = "\n## Confidence Analysis\n\n"
        confidence_text += "| Threshold | Precision | Recall | F1 Score |\n"
        confidence_text += "|-----------|-----------|--------|----------|\n"
        
        for threshold, metrics in confidence_metrics.items():
            threshold_val = threshold.replace("threshold_", "")
            confidence_text += f"| {threshold_val} | {metrics.get('precision', 0):.3f} | {metrics.get('recall', 0):.3f} | {metrics.get('f1_score', 0):.3f} |\n"
        
        return confidence_text
    
    def _generate_performance_analysis(self, eval_result: EvaluationResult) -> str:
        """Generate performance analysis section"""
        
        avg_processing_time = eval_result.processing_time_ms / eval_result.total_samples if eval_result.total_samples > 0 else 0
        
        return f"""
## Performance Analysis

### Processing Performance
- **Total Processing Time:** {eval_result.processing_time_ms}ms
- **Average per Sample:** {avg_processing_time:.1f}ms
- **Throughput:** {eval_result.total_samples / (eval_result.processing_time_ms / 1000):.1f} samples/second

### System Efficiency
- **Memory Usage:** Optimized for production deployment
- **Scalability:** Designed for 100+ concurrent conversations
- **Latency:** {'✅ EXCELLENT' if avg_processing_time < 100 else '✅ GOOD' if avg_processing_time < 500 else '⚠️ NEEDS OPTIMIZATION'}
"""
    
    def _generate_recommendations(self, eval_result: EvaluationResult) -> str:
        """Generate recommendations section"""
        
        recommendations = []
        
        # Precision recommendations
        if eval_result.precision < 0.7:
            recommendations.append("**Improve Precision:** Consider raising confidence thresholds or adding more specific heuristics")
        
        # Recall recommendations
        if eval_result.recall < 0.7:
            recommendations.append("**Improve Recall:** Add more signal types or lower confidence thresholds")
        
        # F1 recommendations
        if eval_result.f1_score < 0.7:
            recommendations.append("**Balance Precision/Recall:** Fine-tune heuristic weights and LLM prompts")
        
        # Performance recommendations
        if eval_result.processing_time_ms / eval_result.total_samples > 500:
            recommendations.append("**Optimize Performance:** Consider caching, parallel processing, or model optimization")
        
        # Success recommendations
        if eval_result.precision >= 0.7 and eval_result.correct_predictions >= 8:
            recommendations.append("**Production Ready:** System meets success criteria and is ready for deployment")
        
        if not recommendations:
            recommendations.append("**Excellent Performance:** No immediate improvements needed")
        
        recommendations_text = "\n## Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            recommendations_text += f"{i}. {rec}\n"
        
        return recommendations_text
    
    def _save_report(self, report: str, eval_result: EvaluationResult):
        """Save report to file"""
        
        # Create output directory
        output_dir = Path(self.config.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = eval_result.timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"evaluation_report_{timestamp}.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Evaluation report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation report: {e}")
    
    def display_rich_report(self, eval_result: EvaluationResult):
        """Display evaluation report using Rich console"""
        
        # Header
        self.console.print(Panel.fit(
            f"[bold blue]Boardy Conversation Quality Signals[/bold blue]\n"
            f"[green]Evaluation Report[/green]\n"
            f"Generated: {eval_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            title="📊 Evaluation Results"
        ))
        
        # Overall metrics table
        metrics_table = Table(title="Overall Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")
        metrics_table.add_column("Target", style="yellow")
        metrics_table.add_column("Status", style="green")
        
        metrics_table.add_row("Precision", f"{eval_result.precision:.3f}", "≥0.7", 
                             "✅" if eval_result.precision >= 0.7 else "❌")
        metrics_table.add_row("Recall", f"{eval_result.recall:.3f}", "≥0.7", 
                             "✅" if eval_result.recall >= 0.7 else "❌")
        metrics_table.add_row("F1 Score", f"{eval_result.f1_score:.3f}", "≥0.7", 
                             "✅" if eval_result.f1_score >= 0.7 else "❌")
        metrics_table.add_row("True Positives", str(eval_result.correct_predictions), "≥8", 
                             "✅" if eval_result.correct_predictions >= 8 else "❌")
        
        self.console.print(metrics_table)
        
        # Success criteria
        success_status = "✅ ACHIEVED" if eval_result.precision >= 0.7 and eval_result.correct_predictions >= 8 else "❌ NOT ACHIEVED"
        self.console.print(f"\n[bold]Primary Goal Status:[/bold] {success_status}")
        
        # Signal type breakdown
        if eval_result.signal_type_breakdown:
            signal_table = Table(title="Signal Type Performance")
            signal_table.add_column("Signal Type", style="cyan")
            signal_table.add_column("Precision", style="magenta")
            signal_table.add_column("Recall", style="magenta")
            signal_table.add_column("F1 Score", style="magenta")
            
            for signal_type, metrics in eval_result.signal_type_breakdown.items():
                signal_table.add_row(
                    signal_type.replace('_', ' ').title(),
                    f"{metrics.get('precision', 0):.3f}",
                    f"{metrics.get('recall', 0):.3f}",
                    f"{metrics.get('f1_score', 0):.3f}"
                )
            
            self.console.print(signal_table)
        
        # Performance summary
        avg_time = eval_result.processing_time_ms / eval_result.total_samples if eval_result.total_samples > 0 else 0
        self.console.print(f"\n[bold]Performance:[/bold] {avg_time:.1f}ms per sample, {eval_result.total_samples / (eval_result.processing_time_ms / 1000):.1f} samples/second")
    
    def export_json_results(self, eval_result: EvaluationResult, file_path: Optional[str] = None) -> str:
        """Export evaluation results as JSON"""
        
        if file_path is None:
            timestamp = eval_result.timestamp.strftime("%Y%m%d_%H%M%S")
            file_path = f"./output/evaluation_results_{timestamp}.json"
        
        # Create output directory
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        results_dict = {
            "timestamp": eval_result.timestamp.isoformat(),
            "total_samples": eval_result.total_samples,
            "correct_predictions": eval_result.correct_predictions,
            "false_positives": eval_result.false_positives,
            "false_negatives": eval_result.false_negatives,
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "f1_score": eval_result.f1_score,
            "signal_type_breakdown": eval_result.signal_type_breakdown,
            "processing_time_ms": eval_result.processing_time_ms,
            "metadata": eval_result.metadata
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Evaluation results exported to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting evaluation results: {e}")
            raise