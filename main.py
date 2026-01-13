import numpy as np
import matplotlib.pyplot as plt

class signal:
    total_signals = 0

    def __init__(self, time_samples, time_unit = "us", signal_unit = None, fs = None):
        self.time_array = np.arange (0,len(time_samples),1) # можно доработать для перехода на другие единицы
        self.time_samples = np.array(time_samples) / 1000000
        self.total_points = len(time_samples)
        self.fs = fs
        signal.total_signals += 1
        self.id = signal.total_signals
        
        if self.fs == None:
            self.freq_step  = 1/(self.total_points) #в МГц
        else:
            self.freq_step  = fs/(self.total_points)

        #self.sample_period = self.total_points #в мкс
        self.freq_array = np.fft.fftfreq(self.total_points, 1) #* 1000000
        self.freq_samples = np.fft.fft(self.time_samples)
        self.freq_abs = np.abs(self.freq_samples)

        print(self.freq_array)
        # print(self.freq_abs)

        

    def plot(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.time_array, self.time_samples, '.')
        plt.title('Time Domain Signal')
        plt.xlabel('Time [us]')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.freq_array, self.freq_abs, '.')
        plt.title('Frequency Domain')
        plt.xlabel('Freq [MHz]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.show()

    def getsamplefreq(self):
        return self.freq_step
    # def fft():
    #     freq.array = np.array()
    #     self.freq_samples = np.fft.fft(self.time_samples) 
    #     return freq_samples
class sinc(signal):
    def __init__(self):
        pass

def main():

    Ts = 2 # Sampling time - us
    N = 255 # Number of points

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
    #print (shape_params["bandwidth"])

    #Интерполляция
    shape_params_inter = {}
    shape_params_inter["name"] = params_str[0]
    shape_params_inter["sample_period"] = int(params_str[1] ) / 1000 # 2ns
    shape_params_inter["total_points"] = int(params_str[2]) * 1000
    shape_params_inter["bandwidth"] = float(params_str[3])
    time_array = np.arange(0,shape_params["total_points"] * shape_params["sample_period"] , shape_params["sample_period"])
    time_array_inter = np.arange(0,shape_params_inter["total_points"] * shape_params_inter["sample_period"] , shape_params_inter["sample_period"])
    time_array_long = time_array_inter
    shape_params_inter["time_values"] = np.interp(time_array_inter,time_array,numbers)

    #Добавление точек для увеличения разрешения по частотного шага
    shape_params_long = {}
    shape_params_long["name"] = params_str[0]
    shape_params_long["sample_period"] = int(params_str[1])
    shape_params_long["total_points"] = int(params_str[2])*1000
    shape_params_long["bandwidth"] = float(params_str[3])
    shape_params_long["time_values"] = numbers + [0] *(255000-255)                 
    #fft
    #shape_params["fft_values"] = np.fft.fft(shape_params["time_values"])
    #shape_params_inter["fft_values"] = np.fft.fft(shape_params_inter["time_values"])
    shape_params_long["fft_values"] = np.fft.fft(shape_params_long["time_values"])
    magnitude = np.abs(shape_params_long["fft_values"])  
    freq_array = 1000 * np.fft.fftfreq(shape_params_long["total_points"], shape_params_long["sample_period"])

    #print (f"sample period: {shape_params_long["sample_period"]} total points: {shape_params_long["total_points"]} ")
    # #Построение временного графика
    # # plt.subplot(2, 1, 1)
    # # plt.plot(time_array, shape_params["time_values"])
    # # plt.title('Time Domain Signal')
    # # plt.xlabel('Time [ms]')
    # # plt.ylabel('Amplitude')
    # # plt.grid(True)

    # plt.subplot(2, 1, 1)
    # plt.plot(time_array_long, shape_params_long["time_values"], '.')
    # plt.title('Time Domain Signal')
    # plt.xlabel('Time [ms]')
    # plt.ylabel('Amplitude')
    # plt.grid(True)

    # #Построение спектра графика
    # #freq_array = np.arange(-1/(2*shape_params["sample_period"]),1/(2*shape_params["sample_period"]), 1/(shape_params["total_points"] * shape_params["sample_period"]))    
    # #freq_array = 1000000 * np.fft.fftfreq(shape_params_inter["total_points"], shape_params_inter["sample_period"])
    
    # #freq_array = np.fft.fftshift(freq_array)
    # plt.subplot(2, 1, 2)
    # plt.plot(freq_array, magnitude, '.')
    # plt.title('Frequency Domain (Magnitude Spectrum)')
    # plt.xlabel('Frequency [kHz]')
    # plt.ylabel('Magnitude')
    # #plt.tight_layout()
    # plt.show()
    # # a = np.array([0,1,2])
    # # b = np.array([2,5,6])
    
    #shape_params
    #Create object signal  
    # sig1 = signal(shape_params["time_values"])
    # sig1.plot()

    sinc_list = [np.sinc(x) for x in np.linspace(-3, 3, 255)]
    sinc_list += 1000 * 254 * [0]
    sig2 = signal(sinc_list)
    #sig2.plot()
    print(f" Шаг по частотуе:{sig2.getsamplefreq():.2e} МГц")


    num_points=1000
    center_freq=0
    width=5
    amplitude=1
    fs=1
    t = np.linspace(-width/2, width/2, num_points)
    print(t)
    # Создание sinc-функции
    # Избегаем деления на 0
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_values = amplitude * np.sinc(t - center_freq)
    
    # Заменяем NaN значения (если center_freq совпадает с каким-то t)
    sinc_values = np.nan_to_num(sinc_values, nan=amplitude)

    plt.subplot(1, 1, 1)
    plt.plot(t, sinc_values, '.')
    plt.title('Sinc')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
       
if __name__ == "__main__":
    main()