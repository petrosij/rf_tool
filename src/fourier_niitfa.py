import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout


with open("wave\\FLASH_shapes1.rfwave","r",encoding="UTF-8") as rffile:
    line1 = rffile.readline()
    lines = rffile.readlines()

#Создание массива точек из файла    
numbers = []
for line in lines:
    numbers.append( int(line.partition(",")[0]) )

#Чтение переменных шейпа
shape_params = {}
params_str = line1.split(",")
shape_params["name"] = params_str[0]
shape_params["sample_period"] = int(params_str[1])
shape_params["total_points"] = int(params_str[2])
shape_params["bandwidth"] = float(params_str[3])
shape_params["time_values"] = numbers



points = shape_params["total_points"]
time = np.linspace(0,510,points)
sig_r = numbers
sig_i = np.zeros(points)

# period = 1
# width11 = 0.11
# width12 = 0.09
# center1 = 0.5
# for i in range(int(points*(center1-width11)), int(points*center1)):
#     part = (i - int(points*(center1-width11)))/(int(points*center1) - int(points*(center1-width11)))
#     sig_r[i] = (np.sin(part*period*np.pi))
# for i in range(int(points*center1), int(points*(center1+width12))):
#     part = (i - int(points*center1))/(int(points*(center1+width12)) - int(points*center1))
#     sig_r[i] = -(np.sin(part*period*np.pi))
#
# period = 1
# width21 = 0.11
# width22 = 0.09
# center2 = 0.5
# for i in range(int(points*(center2-width21)), int(points*center2)):
#     part = (i - int(points*(center2-width21)))/(int(points*center2) - int(points*(center2-width21)))
#     sig_i[i] = -(np.sin(part*period*np.pi))
# for i in range(int(points*center2), int(points*(center2+width22))):
#     part = (i - int(points*center2))/(int(points*(center2+width22)) - int(points*center2))
#     sig_i[i] = -(np.sin(part*period*np.pi))
# """
# sig[int(points*(0.5-width1)) : int(points/2)] = 1
# sig[int(points/2) : int(points*(0.5+width2))] = 1
# """
sig = sig_r + sig_i * 1j

spectrum = np.fft.fftshift(np.fft.fft(sig))
freq = 1000 * np.linspace(-float(0.5*points/(time[-1]-time[0])), float(0.5*points/(time[-1]-time[0])), points)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(time, np.abs(sig))
ax1.set_ylim(bottom=0)
ax1.set_title("Abs signal(t)")
# ax2 = fig.add_subplot(2,1,2)
# ax2.plot(time, sig_r)
# ax2.set_title("Real signal(t)")
# ax3 = fig.add_subplot(2,3,3)
# ax3.plot(time, sig_i)
# ax3.set_title("Imag signal(t)")
ax4 = fig.add_subplot(2,1,2)
ax4.plot(freq, np.abs(spectrum))
ax4.set_ylim(bottom=0)
ax4.set_title("Abs spectrum(f)")
# ax5 = fig.add_subplot(2,3,5)
# ax5.plot(freq, np.real(spectrum))
# ax5.set_title("Real spectrum(f)")
# ax6 = fig.add_subplot(2,3,6)
# ax6.plot(freq, np.imag(spectrum))
# ax6.set_title("Imag spectrum(f)")
fig.tight_layout()
plt.show()

