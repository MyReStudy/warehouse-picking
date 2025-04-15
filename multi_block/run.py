from numpy import load as np_load
from matplotlib import pyplot as plt
from data import generate_data
from functions import load_model
from simple_tsp import am_tsp

plt.rcParams['font.sans-serif'] = ['SimHei']
aisle_num, cross_num = 4,5
goods_low = 44
goods_high = 84
file = 45

def draw_deep(x_coords, y_coords, x_s, y_s):

    coords_list = [(a, b) for a, b in zip(x_coords, y_coords)]
    print(f'路径规划结果：{coords_list}')
    plt.scatter(x_coords, y_coords)
    plt.plot(x_coords, y_coords, '-o', color='green',label='拣货点')
    plt.plot(x_s, y_s, 'x', color='blue',label='交叉点')
    # 标注每个点的坐标
    for i, coord in enumerate(coords_list):
        plt.annotate(f"({coord[0]}, {coord[1]})", xy=coord, xytext=(5, 5), textcoords='offset points')

require_info = np_load('warehouse_data_4_5/required_info.npy', allow_pickle=True)
steiner_info = np_load('warehouse_data_4_5/steiner_info.npy', allow_pickle=True)
model, _ = load_model('pretrained/')

info_stsp = generate_data(require_info,goods_low, goods_high,aisle_num,cross_num)

length, x, y = am_tsp(model, info_stsp, file)
x_s = list(steiner_info[1])
y_s = list(steiner_info[2])
draw_deep(x, y, x_s, y_s)
print(f'路径规划总路程:{length}')

plt.title('AMH-FC拣货节点全连接下求解路径示意图')
caption = "图中拣货点之间连线的距离按照论文中所设定的距离计算方法衡量。"
plt.figlegend([caption], loc='lower center', fontsize=10)
plt.legend()
plt.show()