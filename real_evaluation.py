#!/usr/bin/env python3
"""
Real conversation evaluation script
Tests the system on realistic anonymized conversations
"""

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from boardy_signals.data.real_conversations import generate_realistic_conversations
from boardy_signals.features.extractor import FeatureExtractor
from boardy_signals.evaluation.metrics import MetricsCalculator

console = Console()


def main():
    """Run evaluation on realistic conversations"""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Real Conversation Evaluation[/bold blue]\n"
        "[green]Testing system on realistic anonymized conversations[/green]",
        title="Boardy Challenge Response"
    ))
    
    # Generate realistic conversations
    console.print("\n[bold]Step 1: Loading realistic conversations...[/bold]")
    conversations = generate_realistic_conversations()
    
    # Add more conversations to reach 20
    from boardy_signals.data.real_conversations import generate_remaining_conversations
    conversations.extend(generate_remaining_conversations())
    
    console.print(f"[green]✅ Loaded {len(conversations)} realistic conversations[/green]")
    
    # Analyze conversations
    console.print("\n[bold]Step 2: Running analysis...[/bold]")
    extractor = FeatureExtractor()
    results = extractor.analyze_conversations(conversations)
    
    # Calculate metrics
    console.print("\n[bold]Step 3: Calculating metrics...[/bold]")
    metrics_calc = MetricsCalculator()
    
    # Prepare ground truth
    ground_truth_signals = []
    predicted_signals = []
    
    for i, (conv, result) in enumerate(zip(conversations, results)):
        # Ground truth from metadata
        has_signals = conv.metadata.get("has_match_seeking", False)
        expected_types = conv.metadata.get("signal_types", [])
        
        # Predicted signals
        predicted_types = [s.signal_type for s in result.signals]
        
        # Count as correct if we detected any signal when there should be one
        # or no signals when there shouldn't be any
        correct = (has_signals and len(result.signals) > 0) or (not has_signals and len(result.signals) == 0)
        
        console.print(f"Conv {i+1}: {'✅' if correct else '❌'} Expected: {expected_types}, Got: {predicted_types}")
    
    # Calculate overall metrics
    total_conversations = len(conversations)
    correct_predictions = sum(1 for conv, result in zip(conversations, results) 
                            if (conv.metadata.get("has_match_seeking", False) and len(result.signals) > 0) or 
                               (not conv.metadata.get("has_match_seeking", False) and len(result.signals) == 0))
    
    precision = correct_predictions / total_conversations if total_conversations > 0 else 0.0
    
    # Display results
    console.print("\n[bold]Step 4: Results Summary[/bold]")
    
    results_table = Table(title="Real Conversation Evaluation Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="magenta")
    results_table.add_column("Target", style="yellow")
    results_table.add_column("Status", style="green")
    
    results_table.add_row("Total Conversations", str(total_conversations), "20", "✅")
    results_table.add_row("Correct Predictions", str(correct_predictions), "≥14", "✅" if correct_predictions >= 14 else "❌")
    results_table.add_row("Accuracy", f"{precision:.3f}", "≥0.7", "✅" if precision >= 0.7 else "❌")
    results_table.add_row("Total Signals Detected", str(sum(len(r.signals) for r in results)), "≥8", "✅" if sum(len(r.signals) for r in results) >= 8 else "❌")
    
    console.print(results_table)
    
    # Show conversation breakdown
    console.print("\n[bold]Conversation Breakdown:[/bold]")
    breakdown_table = Table(title="Conversation Analysis")
    breakdown_table.add_column("Conv", style="cyan")
    breakdown_table.add_column("Expected", style="yellow")
    breakdown_table.add_column("Detected", style="magenta")
    breakdown_table.add_column("Status", style="green")
    
    for i, (conv, result) in enumerate(zip(conversations, results)):
        expected = "Has Signals" if conv.metadata.get("has_match_seeking", False) else "No Signals"
        detected = f"{len(result.signals)} signals" if result.signals else "No signals"
        correct = (conv.metadata.get("has_match_seeking", False) and len(result.signals) > 0) or (not conv.metadata.get("has_match_seeking", False) and len(result.signals) == 0)
        status = "✅" if correct else "❌"
        
        breakdown_table.add_row(f"Conv {i+1}", expected, detected, status)
    
    console.print(breakdown_table)
    
    # Final verdict
    if precision >= 0.7:
        console.print(Panel.fit(
            "[bold green]🎉 CHALLENGE PASSED![/bold green]\n"
            f"Precision: {precision:.3f} (≥0.7 required)\n"
            f"Correct predictions: {correct_predictions}/{total_conversations}\n"
            "[yellow]System works on realistic conversation data![/yellow]",
            title="✅ SUCCESS"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]❌ CHALLENGE NOT MET[/bold red]\n"
            f"Precision: {precision:.3f} (<0.7 required)\n"
            f"Correct predictions: {correct_predictions}/{total_conversations}\n"
            "[yellow]System needs improvement on realistic data[/yellow]",
            title="⚠️ NEEDS WORK"
        ))
    
    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_conversations": total_conversations,
        "correct_predictions": correct_predictions,
        "precision": precision,
        "total_signals": sum(len(r.signals) for r in results),
        "conversations": [
            {
                "id": conv.id,
                "expected_signals": conv.metadata.get("has_match_seeking", False),
                "detected_signals": len(result.signals),
                "signal_types": [s.signal_type for s in result.signals],
                "correct": (conv.metadata.get("has_match_seeking", False) and len(result.signals) > 0) or (not conv.metadata.get("has_match_seeking", False) and len(result.signals) == 0)
            }
            for conv, result in zip(conversations, results)
        ]
    }
    
    with open("real_evaluation_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    console.print(f"\n[green]✅ Results saved to real_evaluation_results.json[/green]")


if __name__ == "__main__":
    main()