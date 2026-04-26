import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy import signal
from scipy.ndimage import label, center_of_mass

# ============================================================================
# PROFESSIONAL OFDM ISAC RADAR SYSTEM
# ============================================================================
# Version: 6.0 (Integrated Production Edition)
# Author: Professional DSP/Radar Systems Engineer
# ============================================================================

# Physical Constants
C = 3e8                    # Speed of light (m/s)
BOLTZMANN = 1.38e-23      # Boltzmann constant (J/K)
T0 = 290                   # Reference temperature (K)

# --- System Configuration ---
FS = 20e6                  # Sampling frequency (20 MHz)
FC = 2.4e9                 # Carrier frequency (2.4 GHz - WiFi/ISM)
LAMBDA = C / FC            # Wavelength (m)

# Waveform Parameters
PRI = 250e-6               # Pulse Repetition Interval (250 us -> 4 kHz PRF)
N_PULSES = 512             # Coherent Processing Interval (128 ms)
N_SC = 512                 # OFDM subcarriers
CP = 256                   # Cyclic Prefix (covers up to 1.9 km)
N_SAMPLES_PER_PRI = N_SC + CP

# Receiver Characteristics
TX_POWER_DBM = 30          # Transmit power (30 dBm = 1W)
TX_GAIN_DB = 13            # Transmit antenna gain (dBi)
RX_GAIN_DB = 13            # Receive antenna gain (dBi)
NOISE_FIGURE_DB = 6        # Receiver noise figure (dB)
THERMAL_NOISE_DBM = -174   # dBm/Hz

# Calculated Noise Floor
NOISE_FLOOR_DBM = THERMAL_NOISE_DBM + 10*np.log10(FS) + NOISE_FIGURE_DB
NOISE_FLOOR_W = 10**((NOISE_FLOOR_DBM - 30) / 10)

def radar_range_equation(range_m, rcs_dbsm):
    """Calculates received power based on physical radar equation."""
    p_tx = 10**((TX_POWER_DBM - 30) / 10)
    g_tx = 10**(TX_GAIN_DB / 10)
    g_rx = 10**(RX_GAIN_DB / 10)
    rcs = 10**(rcs_dbsm / 10)
    
    # Pr = (Pt * Gt * Gr * lambda^2 * sigma) / ((4 * pi)^3 * R^4)
    num = p_tx * g_tx * g_rx * (LAMBDA**2) * rcs
    den = ((4 * np.pi)**3) * (range_m**4)
    return num / den

def generate_ofdm_pulse():
    """Generates a single OFDM pulse with QPSK modulation, normalized to unit power."""
    data = (np.random.choice([-1, 1], N_SC) + 1j * np.random.choice([-1, 1], N_SC)) / np.sqrt(2)
    time_data = ifft(data)
    # Normalize such that mean power is 1
    time_data /= np.sqrt(np.mean(np.abs(time_data)**2))
    cp_data = time_data[-CP:]
    return np.concatenate([cp_data, time_data])

def simulate_environment(targets):
    """Simulates the radar environment including targets, leakage, and noise."""
    ref_pulses = []
    surv_pulses = []
    
    t_fast = np.arange(N_SAMPLES_PER_PRI) / FS
    
    for p in range(N_PULSES):
        # Generate Transmit Signal
        tx_sig = generate_ofdm_pulse()
        ref_pulses.append(tx_sig)
        
        # Initialize Surveillance Signal
        surv_sig = np.zeros(N_SAMPLES_PER_PRI, dtype=complex)
        
        # 1. Direct Path Leakage (Strong interference)
        # Typical isolation is 40-60 dB
        leakage_gain = 10**(-45/20) 
        surv_sig += tx_sig * leakage_gain
        
        # 2. Target Echoes
        for tgt in targets:
            range_m = tgt['R']
            velocity = tgt['v']
            rcs = tgt['rcs']
            
            # Time delay and Doppler shift
            delay = 2 * range_m / C
            fd = 2 * velocity / LAMBDA
            
            # Slow-time phase rotation
            phase_slow = np.exp(1j * 2 * np.pi * fd * p * PRI)
            
            # Fast-time Doppler and delay (Fractional Delay implementation)
            tx_fft = fft(tx_sig)
            freqs = fftfreq(N_SAMPLES_PER_PRI, d=1/FS)
            shifted_fft = tx_fft * np.exp(-1j * 2 * np.pi * freqs * delay)
            
            # Apply Doppler frequency shift in time domain for accuracy
            echo_time = ifft(shifted_fft)
            echo_time *= np.exp(1j * 2 * np.pi * fd * t_fast)
            
            # Amplitude based on Radar Equation
            p_rx = radar_range_equation(range_m, rcs)
            echo_amp = np.sqrt(p_rx)
            
            surv_sig += echo_time * phase_slow * echo_amp
            
        # 3. Add Thermal Noise
        noise = np.sqrt(NOISE_FLOOR_W / 2) * (np.random.randn(N_SAMPLES_PER_PRI) + 1j * np.random.randn(N_SAMPLES_PER_PRI))
        surv_sig += noise
        
        surv_pulses.append(surv_sig)
        
    return np.array(ref_pulses), np.array(surv_pulses)

def process_signal(ref, surv):
    """Core DSP pipeline: Range-Doppler processing for OFDM."""
    # 1. Range Compression (using standard OFDM division method)
    # Extract the symbol part (removing CP)
    ref_sym = ref[:, CP:]
    surv_sym = surv[:, CP:]
    
    # FFT to frequency domain
    ref_freq = fft(ref_sym, axis=1)
    surv_freq = fft(surv_sym, axis=1)
    
    # Element-wise division to remove data modulation
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    div_freq = surv_freq / (ref_freq + eps)
    
    # IFFT back to time domain to get range profiles
    range_profiles = ifft(div_freq, axis=1)
    
    # 2. MTI Filtering (Moving Target Indicator)
    # Simple DC subtraction to remove stationary clutter/leakage 
    # while preserving slow moving targets.
    mti_output = range_profiles - np.mean(range_profiles, axis=0)
    
    # 3. Doppler Processing (Slow-time FFT)
    n_pulses_processed, n_range = mti_output.shape
    doppler_window = np.hamming(n_pulses_processed)[:, np.newaxis]
    rd_map = fftshift(fft(mti_output * doppler_window, axis=0), axes=0)
    
    return rd_map

def detect_targets(rd_map, pfa=1e-6):
    """Adaptive 2D CFAR detection and clustering."""
    power_map = np.abs(rd_map)**2
    n_doppler, n_range = power_map.shape
    
    # CFAR Parameters
    guard = 2
    train = 8
    win_size = 2 * (guard + train) + 1
    
    # Construct Kernel
    kernel = np.ones((win_size, win_size))
    kernel[train:train+2*guard+1, train:train+2*guard+1] = 0
    n_train = np.sum(kernel)
    
    # Compute adaptive threshold
    noise_est = signal.convolve2d(power_map, kernel/n_train, mode='same', boundary='wrap')
    alpha = n_train * (pfa**(-1/n_train) - 1)
    threshold = alpha * noise_est
    
    detections = power_map > threshold
    
    # Clustering (DBSCAN-lite using connected components)
    labeled, num_features = label(detections)
    found_targets = []
    
    range_res = C / (2 * FS)
    # Note: No pulses lost in DC subtraction
    n_pulses_processed = rd_map.shape[0]
    vel_res = LAMBDA / (2 * n_pulses_processed * PRI)
    
    v_axis = (np.arange(n_doppler) - n_doppler//2) * vel_res
    r_axis = np.arange(n_range) * range_res
    
    for i in range(1, num_features + 1):
        mask = (labeled == i)
        # Centroid weighting by power
        coords = np.argwhere(mask)
        weights = power_map[mask]
        
        # Weighted average for sub-bin accuracy
        idx_v = np.average(coords[:, 0], weights=weights)
        idx_r = np.average(coords[:, 1], weights=weights)
        
        v_est = np.interp(idx_v, np.arange(n_doppler), v_axis)
        r_est = np.interp(idx_r, np.arange(n_range), r_axis)
        
        peak_snr = 10 * np.log10(np.max(weights) / np.mean(noise_est[mask]))
        
        found_targets.append({'R': r_est, 'v': v_est, 'SNR': peak_snr})
        
    return detections, found_targets

def plot_results(rd_map, detections, found_targets, true_targets):
    """Comprehensive visualization of radar performance."""
    n_doppler, n_range = rd_map.shape
    range_res = C / (2 * FS)
    vel_res = LAMBDA / (2 * n_doppler * PRI)
    
    v_extent = [-(n_doppler//2)*vel_res, (n_doppler//2)*vel_res]
    r_extent = [0, n_range*range_res]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Range-Doppler Heatmap
    plt.subplot(2, 2, 1)
    rd_db = 10 * np.log10(np.abs(rd_map)**2 + 1e-15)
    plt.imshow(rd_db, aspect='auto', extent=[r_extent[0], r_extent[1], v_extent[0], v_extent[1]], cmap='jet')
    plt.colorbar(label='Power (dB)')
    plt.title('Range-Doppler Map (MTI Applied)')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    
    # 2. Detection Map
    plt.subplot(2, 2, 2)
    plt.imshow(detections, aspect='auto', extent=[r_extent[0], r_extent[1], v_extent[0], v_extent[1]], cmap='gray')
    plt.title('CFAR Detections')
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    
    # 3. Range Profile (at Peak Doppler of first target if exists)
    plt.subplot(2, 2, 3)
    if found_targets:
        # Find best target peak
        idx = np.argmax(np.abs(rd_map))
        v_idx, r_idx = np.unravel_index(idx, rd_map.shape)
        plt.plot(np.arange(n_range)*range_res, rd_db[v_idx, :])
        plt.title(f'Range Profile at v = {v_extent[0] + v_idx*vel_res:.2f} m/s')
    else:
        plt.plot(np.arange(n_range)*range_res, rd_db[n_doppler//2, :])
        plt.title('Range Profile at Zero Doppler')
    plt.grid(True)
    plt.xlabel('Range (m)')
    plt.ylabel('Intensity (dB)')
    
    # 4. Target Report
    plt.subplot(2, 2, 4)
    plt.axis('off')
    report = "TARGET REPORT\n\n"
    report += f"{'Type':<10} | {'Range (m)':<10} | {'Vel (m/s)':<10} | {'SNR (dB)':<8}\n"
    report += "-"*50 + "\n"
    for i, t in enumerate(true_targets):
        report += f"TRUE {i+1:<5} | {t['R']:<10.1f} | {t['v']:<10.1f} | {'N/A':<8}\n"
    report += "-"*50 + "\n"
    for i, t in enumerate(found_targets):
        report += f"EST  {i+1:<5} | {t['R']:<10.1f} | {t['v']:<10.1f} | {t['SNR']:<8.1f}\n"
    
    plt.text(0, 1, report, family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Define Realistic Targets
    # Drone: RCS ~ -20 dBsm
    # Car: RCS ~ 10 dBsm
    # Human: RCS ~ 0 dBsm
    test_targets = [
        {'R': 150, 'v': 15,  'rcs': -15}, # Fast moving drone
        {'R': 400, 'v': -5,  'rcs': 0},   # Human walking away
        {'R': 800, 'v': 25,  'rcs': 15},  # Distant car
    ]
    
    print("Simulating Environment...")
    ref, surv = simulate_environment(test_targets)
    
    print("Processing Signal...")
    rd_map = process_signal(ref, surv)
    
    print("Running CFAR Detection...")
    detections, found = detect_targets(rd_map, pfa=1e-6)
    
    print(f"Found {len(found)} targets.")
    for f in found:
        print(f"  Target at R={f['R']:.1f}m, v={f['v']:.1f}m/s, SNR={f['SNR']:.1f}dB")
        
    plot_results(rd_map, detections, found, test_targets)
