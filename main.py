import numpy as np
import matplotlib.pyplot as plt

def interpolate_shape_linear(shape_points, num_new_points):
    """
    Линейная интерполяция точек формы
    
    Parameters:
    -----------
    shape_points : array-like, shape (N, 2)
        Исходные точки формы [x, y]
    num_new_points : int
        Количество точек после интерполяции
    
    Returns:
    --------
    interpolated_points : ndarray, shape (num_new_points, 2)
        Интерполированные точки
    """
    # Вычисляем кумулятивное расстояние вдоль кривой
    distances = np.zeros(len(shape_points))
    for i in range(1, len(shape_points)):
        dx = shape_points[i, 0] - shape_points[i-1, 0]
        dy = shape_points[i, 1] - shape_points[i-1, 1]
        distances[i] = distances[i-1] + np.sqrt(dx**2 + dy**2)
    
    # Нормализуем расстояния к [0, 1]
    distances_normalized = distances / distances[-1]
    
    # Параметризация
    t_original = distances_normalized
    t_new = np.linspace(0, 1, num_new_points)
    
    # Интерполяция по каждой координате
    x_interp = np.interp(t_new, t_original, shape_points[:, 0])
    y_interp = np.interp(t_new, t_original, shape_points[:, 1])
    
    return np.column_stack([x_interp, y_interp])

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
    shape_params_inter["time_values"] = np.interp(time_array_inter,time_array,numbers)


    #fft
    #shape_params["fft_values"] = np.fft.fft(shape_params["time_values"])
    shape_params_inter["fft_values"] = np.fft.fft(shape_params_inter["time_values"])
    magnitude = np.abs(shape_params_inter["fft_values"])  


    #Построение временного графика
    # plt.subplot(2, 1, 1)
    # plt.plot(time_array, shape_params["time_values"])
    # plt.title('Time Domain Signal')
    # plt.xlabel('Time [ms]')
    # plt.ylabel('Amplitude')
    # plt.grid(True)

    plt.subplot(2, 1, 1)
    plt.plot(time_array_inter, shape_params_inter["time_values"], '.')
    plt.title('Time Domain Signal')
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    #Построение спектра графика
    #freq_array = np.arange(-1/(2*shape_params["sample_period"]),1/(2*shape_params["sample_period"]), 1/(shape_params["total_points"] * shape_params["sample_period"]))    
    freq_array = 1000000 * np.fft.fftfreq(shape_params_inter["total_points"], shape_params_inter["sample_period"])
    #freq_array = np.fft.fftshift(freq_array)
    plt.subplot(2, 1, 2)
    plt.plot(freq_array, magnitude, '.')
    plt.title('Frequency Domain (Magnitude Spectrum)')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Magnitude')
    #plt.tight_layout()
    plt.show()
    # a = np.array([0,1,2])
    # b = np.array([2,5,6])
    
    #shape_params



    
if __name__ == "__main__":
    main()