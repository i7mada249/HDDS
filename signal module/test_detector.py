import sys
import os
import numpy as np

# Add the current directory to path so we can import our new module
sys.path.append(os.getcwd())

from professional_isac_detector import simulate_environment, process_signal, detect_targets

def test_detection_performance():
    # Define targets
    test_targets = [
        {'R': 200, 'v': 10,  'rcs': 0},   # Human at 200m
        {'R': 500, 'v': -20, 'rcs': 10},  # Car at 500m
    ]
    
    print("Starting simulation...")
    ref, surv = simulate_environment(test_targets)
    
    print("Processing...")
    rd_map = process_signal(ref, surv)
    
    print("Detecting...")
    detections, found = detect_targets(rd_map, pfa=1e-3)
    
    power_map = np.abs(rd_map)**2
    max_idx = np.unravel_index(np.argmax(power_map), power_map.shape)
    print(f"Power map max: {np.max(power_map):.2e} at {max_idx}")
    print(f"Power map mean: {np.mean(power_map):.2e}")
    
    # Expected locations
    # Target 1: R=200, v=10
    range_res = 3e8 / (2 * 20e6)
    # vel_res calculation depends on n_doppler
    n_doppler = rd_map.shape[0]
    vel_res = (3e8/5.8e9) / (2 * n_doppler * 1e-3)
    r_idx = int(200 / range_res)
    v_idx = int(n_doppler//2 + 10 / vel_res)
    
    if 0 <= v_idx < rd_map.shape[0] and 0 <= r_idx < rd_map.shape[1]:
        print(f"Power at target 1 ([{v_idx}, {r_idx}]): {power_map[v_idx, r_idx]:.2e}")
        print(f"Relative to mean: {10*np.log10(power_map[v_idx, r_idx]/np.mean(power_map)):.1f} dB")
    
    # Check zero doppler (leakage residue)
    v_zero = rd_map.shape[0] // 2
    print(f"Power at zero Doppler, range 0: {power_map[v_zero, 0]:.2e}")
    print(f"Max power in zero Doppler bin: {np.max(power_map[v_zero, :]):.2e}")
    
    print(f"\nDetection Results (Total {len(found)}):")
    for f in found:
        print(f"  Detected: R={f['R']:.1f}m, v={f['v']:.1f}m/s, SNR={f['SNR']:.1f}dB")
    
    # Simple validation
    if len(found) >= 2:
        print("\nSUCCESS: At least 2 targets detected.")
    else:
        print("\nFAILURE: Did not detect all targets.")

if __name__ == "__main__":
    test_detection_performance()
