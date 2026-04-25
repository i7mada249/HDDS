import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- إعدادات الإشارة ---
FS = 1.92e6  
N_FFT = 128   
CP_LEN = 9    

def generate_lte_symbol():
    # توليد بيانات عشوائية QAM-16
    bits = np.random.randint(0, 16, N_FFT)
    qam = (np.take([-3, -1, 1, 3], bits >> 2) + 
           1j * np.take([-3, -1, 1, 3], bits & 3))
    
    # تحويل من تردد إلى زمن
    time_signal = np.fft.ifft(qam)
    
    # إضافة المدى الدوري
    cp = time_signal[-CP_LEN:]
    return np.concatenate([cp, time_signal])

# إعداد الشكل والمحاور
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# الرسم الأول: النطاق الزمني
line1, = ax1.plot([], [], lw=1, color='#1f77b4')
ax1.set_title("إشارة 4G (LTE) في نطاق الزمن - Time Domain")
ax1.set_xlim(0, N_FFT + CP_LEN)
ax1.set_ylim(-0.5, 0.5)
ax1.grid(True, alpha=0.3)

# الرسم الثاني: نطاق التردد (Stem Plot)
# ننشئ Stem plot ابتدائي
x_freq = np.arange(N_FFT)
markerline, stemlines, baseline = ax2.stem(x_freq, np.zeros(N_FFT), basefmt=" ")
ax2.set_title("توزيع الترددات الحاملة - Frequency Spectrum (OFDM Subcarriers)")
ax2.set_xlim(0, N_FFT)
ax2.set_ylim(0, 15)
ax2.grid(True, alpha=0.3)

def init():
    line1.set_data([], [])
    return line1, markerline

def update(frame):
    # 1. توليد الرمز
    symbol = generate_lte_symbol()
    
    # 2. تحديث رسم الزمن
    line1.set_data(np.arange(len(symbol)), symbol.real)
    
    # 3. تحديث رسم التردد
    freqs = np.abs(np.fft.fft(symbol[CP_LEN:]))**2
    
    # تحديث النقاط (Markers) - الطريقة المتوافقة مع Line2D
    markerline.set_data(x_freq, freqs)
    
    # تحديث الخطوط العمودية (Segments)
    # كل خط عبارة عن مصفوفة [[x, 0], [x, y]]
    segments = [np.array([[i, 0], [i, f]]) for i, f in enumerate(freqs)]
    stemlines.set_segments(segments)
    
    return line1, markerline, stemlines

# تقليل interval لزيادة سرعة التحديث (Real-time feeling)
ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=True, interval=30)

plt.tight_layout()
plt.show()