import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.getcwd())

from professional_isac_detector import simulate_environment, process_signal, detect_targets, plot_results

def run_scenario(name, targets):
    print(f"\n{'='*60}")
    print(f" SCENARIO: {name}")
    print(f"{'='*60}")
    
    print("Simulating...")
    ref, surv = simulate_environment(targets)
    
    print("Processing...")
    rd_map = process_signal(ref, surv)
    
    print("Detecting...")
    detections, found = detect_targets(rd_map, pfa=1e-6)
    
    print(f"Found {len(found)} targets.")
    for f in found:
        print(f"  Target: R={f['R']:.1f}m, v={f['v']:.1f}m/s, SNR={f['SNR']:.1f}dB")
        
    # In a real environment we would show plots, 
    # but here we just confirm detection of true targets
    for t in targets:
        success = any(abs(f['R'] - t['R']) < 15 and abs(f['v'] - t['v']) < 1 for f in found)
        status = "✅ MATCHED" if success else "❌ MISSED"
        print(f"  Ground Truth: R={t['R']}m, v={t['v']}m/s -> {status}")

if __name__ == "__main__":
    scenarios = [
        {
            "name": "Clear Sky (Noise Floor Validation)",
            "targets": []
        },
        {
            "name": "Slow Drone (Approaching)",
            "targets": [{'R': 800, 'v': 5, 'rcs': -20}]
        },
        {
            "name": "Fast Vehicle (High Doppler)",
            "targets": [{'R': 1200, 'v': -35, 'rcs': 10}]
        },
        {
            "name": "Multi-Target Swarm",
            "targets": [
                {'R': 300, 'v': 10, 'rcs': -15},
                {'R': 315, 'v': 12, 'rcs': -15},
                {'R': 600, 'v': -8, 'rcs': -10}
            ]
        },
        {
            "name": "Low SNR Detection Limit",
            "targets": [{'R': 1500, 'v': 3, 'rcs': -25}]
        }
    ]
    
    for sc in scenarios:
        run_scenario(sc['name'], sc['targets'])
