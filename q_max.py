
import matplotlib.pyplot as plt
import matplotlib

size = 12

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : size})
matplotlib.rcParams['axes.linewidth'] = 1.

with open(r'D:\LaserYb\Medidas Espectrometro\17_02_2023\max1.txt') as data:
    content = data.read()
    content_list = list(map(float, content[:-1].split('\n')))
    
    current = [(410 - x * 0.8) for x in range(0, 214, 2)]
    current.reverse()
    indexes = [75,73,71,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
    for i in indexes:
        del content_list[i]
        del current[i]

    fig, ax = plt.subplots()
    ax.plot(current, content_list, '-',lw=1, c='firebrick', zorder=1)
    ax.scatter(current, content_list, s=30, facecolors='white', edgecolors='firebrick', zorder=3)
    ax.tick_params(axis='both', which='major', labelsize=size)
    ax.set_xlabel('Current (mA)', labelpad=15)
    ax.set_ylabel(r'$|q|$', labelpad=15)
    plt.tight_layout()
    plt.show()

    