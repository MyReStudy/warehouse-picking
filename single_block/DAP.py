import ast
import math
import PickingInfo
from PickingInfo import *
import time

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)


class DAP:

    def __init__(self, distanceMatrixPath, warehouseInformation):
        self.distanceMatrix = pd.read_excel(distanceMatrixPath, index_col=0)  # 距离矩阵 记录每一个拣货巷道中交叉点和拣货点之间的距离 以及拣货巷道间交叉点的距离
        self.warehouseInformation = pd.read_excel(warehouseInformation, index_col=0)  # 仓库信息 记录每个拣货点存在哪个巷道
        self.aisleList = getAisleList(self.warehouseInformation)  # 从小到大排序的拣货巷道列表 已经去除了空的
        self.aisleLength = self.distanceMatrix['a{}'.format(self.aisleList[0])]['b{}'.format(self.aisleList[0])]  # 由于是等长 所以这里为拣货巷道长度
        self.result = initResult(self.aisleList)  # 初始化结果表
        self.calNum=0
        self.stateList=PickingInfo.stateList  # 状态state名称的列表
        self.cofList=PickingInfo.cofList  # 非空巷道的配置类型 1:(i) 2:(ii) 3:(iii) 4:(iv)
        self.calNum=0
        self.startTime=0.0
        self.endTime = 0.0
        self.finalResult=0
        self.totalTime=0.0
        # 注:由于删去了空列 所以cross的长度不等长

    def findLargestGap(self, vertexContainedInAisle,aisle):
        '''
        :param vertexContainedInAisle: 想找的那个巷道所有的点
        :param aisle: 巷道
        :return: 上面的点和下面的点
        '''
        maximumSpac = 0
        distanceFromA = dict.fromkeys(vertexContainedInAisle)
        vertex_i = ''
        vertex_j = ''
        for v in vertexContainedInAisle:
            distanceFromA[v] = self.distanceMatrix.loc['a{}'.format(aisle)][v]  # a到各点的距离
        distanceFromASorted = sorted(distanceFromA.items(), key=lambda x: x[1], reverse=False)  # 根据值排序
        for i in range(0, len(distanceFromASorted) - 1):
            vertex1 = distanceFromASorted[i][0]
            vertex2 = distanceFromASorted[i + 1][0]
            if self.distanceMatrix[vertex1][vertex2] > maximumSpac:
                maximumSpac = self.distanceMatrix[vertex1][vertex2]
                vertex_i = vertex1
                vertex_j = vertex2
        if self.distanceMatrix['a{}'.format(aisle)][vertex_i] <= self.distanceMatrix['a{}'.format(aisle)][
            vertex_j]:
            topVertex = vertex_i
            downVertex = vertex_j
        else:
            topVertex = vertex_j
            downVertex = vertex_i
        return topVertex,downVertex

    def findCrossVertex(self, vertexContainedInAisle, aisle):
        '''
        找间距最大的两个点
        :param vertexContainedInAisle: 想找的那个巷道所有的点
        :param aisle: 巷道
        :return: 上面的点和下面的点
        '''
        maximumSpac = 0
        distanceFromA = dict.fromkeys(vertexContainedInAisle)
        vertex_i = ''
        vertex_j = ''
        for v in vertexContainedInAisle:
            distanceFromA[v] = self.distanceMatrix.loc['a{}'.format(aisle)][v]  # 上交叉点到各点的距离
        distanceFromASorted = sorted(distanceFromA.items(), key=lambda x: x[1], reverse=False)  # 根据值排序
        for i in range(0, len(distanceFromASorted) - 1):
            vertex1 = distanceFromASorted[i][0]
            vertex2 = distanceFromASorted[i + 1][0]
            if self.distanceMatrix[vertex1][vertex2] >= maximumSpac: # 如果这两个点的距离大于等于最大间隔
                maximumSpac = self.distanceMatrix[vertex1][vertex2]  # 最大间隔等于这两个点之间的距离
                vertex_i = vertex1  # 记录最大间隔的两个点
                vertex_j = vertex2
        if self.distanceMatrix['a{}'.format(aisle)][vertex_i] <= self.distanceMatrix['a{}'.format(aisle)][vertex_j]:  # 判断这两个点的上下位置关系
            topVertex = vertex_i
            downVertex = vertex_j
        else:
            topVertex = vertex_j
            downVertex = vertex_i
        return topVertex,downVertex

    def calculateCross(self, stageFrom, stageTo):
        '''
        计算横向通道距离
        :param stageFrom: 从哪个拣货巷道来
        :param stageTo: 到哪个拣货巷道去
        :return: 横向通道长度
        '''
        lengthForState={}
        for state in self.stateList:  # 计算每个状态state带来的横向巷道产生的距离
            if state == 'state1':
                lengthForState[state]=(self.distanceMatrix['a{}'.format(stageFrom)]['a{}'.format(stageTo)] +
                                      self.distanceMatrix['b{}'.format(stageFrom)]['b{}'.format(stageTo)])
            if state == 'state2':
                lengthForState[state]=(self.distanceMatrix['a{}'.format(stageFrom)]['a{}'.format(stageTo)] * 2)
            if state == 'state3':
                lengthForState[state]=(self.distanceMatrix['b{}'.format(stageFrom)]['b{}'.format(stageTo)] * 2)
            if state == 'state4_1':
                lengthForState[state] = (2 * (self.distanceMatrix['a{}'.format(stageFrom)]['a{}'.format(stageTo)] + self.distanceMatrix['b{}'.format(stageFrom)]['b{}'.format(stageTo)]))
            if state == 'state4_2':
                lengthForState[state] = lengthForState['state4_1']
        return lengthForState

    def calculationVertical(self, aisle, aisleLength):
        """
        计算竖向通道距离
        :param configuration: 竖向的配置方法
        :param aisle: 竖向的通道编号
        :return: 竖向的总距离
        """
        lengthForCof={}
        vertexContainedInAisle = self.warehouseInformation[self.warehouseInformation.aisle == aisle].index.tolist()  # 取出包含在该拣货巷道里面的vertex
        distanceInAisle = []
        for vertex in vertexContainedInAisle:
            distanceInAisle.append(self.distanceMatrix['a{}'.format(aisle)][vertex])  # 获取该巷道内上交叉点到各个拣货点的距离
        maxIndexA = distanceInAisle.index(max(distanceInAisle))  # 取到与上拣货点距离最远的点的index
        maxDistanceA = self.distanceMatrix['a{}'.format(aisle)][vertexContainedInAisle[maxIndexA]]  # 取到与上拣货点最远的点对应的最大距离
        maxIndexB = distanceInAisle.index(min(distanceInAisle))  # 取到与下拣货点距离最远的点的index（即与上拣货点距离最近的点）
        maxDistanceB = self.distanceMatrix['b{}'.format(aisle)][vertexContainedInAisle[maxIndexB]]  # 取到与下拣货点最远的点对应的最大距离
        for cof in self.cofList:  # 计算该巷道内不同的模式带来的距离
            if cof == 1:
                lengthForCof[cof]=aisleLength
            elif cof == 2:
                lengthForCof[cof]=maxDistanceA*2
            elif cof == 3:
                lengthForCof[cof]=maxDistanceB * 2
            else:  # 如果是模式(iv)，则需要判断巷道内点的数量
                if len(vertexContainedInAisle) <= 1:  # 如果点的数量小于等于1个
                    lengthForCof[cof]=None
                else:  # 否则找到最大间隔
                    topVertex, downVertex = self.findCrossVertex(vertexContainedInAisle, aisle)  # 获取最大间隔
                    distanceInAisle = self.distanceMatrix['a{}'.format(aisle)][topVertex] * 2 + self.distanceMatrix['b{}'.format(aisle)][downVertex] * 2
                    lengthForCof[cof]=distanceInAisle

        return lengthForCof

    def formingPath(self):
        """
        主方法 生成路径
        """
        aisleList = self.aisleList  # 获取所有的aisle
        if len(aisleList)==1:  # 只有一个巷道
            distance = self.calculationVertical(aisleList[0],self.aisleLength)
            del distance[1]
            del distance[4]
            best = sorted(distance.items(), key=lambda x: x[1])
            return best[0][0], best[0][1], -1
        else:
            result = self.result  # 结果表
            crossDict0 = self.calculateCross(aisleList[0], aisleList[1])  # 计算每个状态带来的横向距离
            verDict0 = self.calculationVertical(1, self.aisleLength)  # 计算每个拣货巷道内不同模式的路程
            for state in stateTransfer.keys(): #先处理初始状态
                if state == 'state4_1':  # 初始状态不出现state4_1
                    result.loc[state, 1] = None
                else:
                    if verDict0[initAndLastState[state]] is not None:
                        distance = crossDict0[state]+verDict0[initAndLastState[state]]  # 获取当前状态下的横向巷道长度 可能的配置对应的距离长度 相加
                        if distance:
                            result.loc[state, 1] = '{},{}'.format([state, initAndLastState[state]], distance)  # 记录下来第一条巷道每个状态对应的路程
                    else:
                        result.loc[state, 1] = None
            for i in range(1, len(aisleList)-1):  # i是aisle列表的下标
                currentAisle = aisleList[i]  # 当前拣货巷道
                preAisle = aisleList[i - 1]  # 上一个拣货巷道
                nextAisle = aisleList[i+1]  # 下一个拣货巷道
                crossDist=self.calculateCross(currentAisle,nextAisle) # 横向距离列表
                verDist=self.calculationVertical(currentAisle,self.aisleLength)
                for state in stateTransfer.keys():  # 设定当前状态 当当前状态和上一阶段状态的配置方式确定后，中间巷道的配置是唯一确定的（取最小的）
                    transferMethod = stateTransfer[state]  # 获取当前状态可能的转移方式
                    distanceList = []
                    methodList = []
                    for preState in transferMethod:  # 遍历所有可能转移方式中对应的上一个阶段的状态
                        if result.loc[preState, preAisle] is None:  # 如果不能通过这种方式状态转移
                            pass
                        else:
                            config = transferMethod[preState]  # 拣货巷道可能的配置
                            if (type(config).__name__=='list'):
                                valueList=[]
                                for c in config:
                                    if verDist[c] is not None:  # 如果拣货巷道可以是该配置
                                        valueList.append(verDist[c])  # 将该配置对应的路程长度存起来
                                    else:
                                        valueList.append(math.inf)  # 否则存储无穷大（即不可能有该配置）
                                minConfig = config[valueList.index(min(valueList))]  # 拣货巷道对应路径最短的配置方式
                            else:
                                minConfig = config  # 如果只有一种配置方式 直接作为产生路径最短的配置方式记录
                            preDistance = float(result.loc[preState, preAisle].split(',')[-1])  # 指标函数值 前一个状态下的路程
                            if verDist[minConfig] is not None:
                                distance = float(preDistance) + float(verDist[minConfig])  # 将当前最短的拣货巷道配置方式累加到历史距离中

                                distanceList.append(distance)
                                methodList.append((preState,minConfig))
                    points = zip(distanceList, methodList)
                    points = sorted(points)  # 按照距离由小到大排序（距离，本阶段的决策）
                    self.result.loc[state, currentAisle] = '{},{}'.format(points[0][1], points[0][0]+float(crossDist[state]))  # 记录决策与距离
            # 最后一个阶段
            verDictn = self.calculationVertical(aisleList[-1], self.aisleLength)  # 计算最后一个拣货巷道不同的模式以及对应的距离
            for state in stateTransfer.keys():
                if self.result.loc[state, aisleList[-2]]:
                    distance = verDictn[initAndLastState[state]]
                    if distance is not None:
                        self.result.loc[state, self.aisleList[-1]] = '{},{}'.format([initAndLastState[state]], float(self.result.loc[state, aisleList[-2]].split(',')[-1]) + distance)  # 记录决策与距离
                    else:
                        self.result.loc[state, self.aisleList[-1]] = None
            # 往前回溯 得到最优路径对应的配置方式
            path = []
            result_last_aisle = result.iloc[:-1, -1]
            temp = []
            config = []
            for i in result_last_aisle:
                if not pd.isna(i):
                    temp.append(float(i.split(',')[-1]))
                    config.append(i.split(',')[0])
            idx = temp.index(min(temp))  # 路径最短的状态对应的index
            stateList = ['state1', 'state2', 'state3', 'state4_1', 'state4_2']
            state_last = stateList[idx]
            config_last = config[idx]
            path.append(config_last)  # 最后一个拣货巷道的配置
            columns_to_keep = result.columns[:-1].tolist()
            # 使用新的列名列表来选择DataFrame中的列
            result_without_last_column = result[columns_to_keep]
            columns = result_without_last_column.columns.tolist()
            reversed_columns = columns[::-1]
            for column in reversed_columns:
                # 获取值并转换为字符串（如果它已经是字符串，则这一步是多余的）
                value = str(result_without_last_column.loc[state_last, column])  # 回溯
                part_before_last_comma = ','.join(value.split(',')[:-1])
                path.append(part_before_last_comma)
                tuple_tmp = ast.literal_eval(part_before_last_comma)
                state_last = tuple_tmp[0]  # 更新状态 继续回溯
        return stateList[idx], temp[idx], path

    def start(self):
        self.startTime=time.perf_counter()

    def end(self):
        self.endTime=time.perf_counter()
        self.totalTime=self.endTime-self.startTime

if __name__ == '__main__':
    distanceMatrixPath = 'data/距离矩阵.xlsx'
    warehouseInformation = 'data/仓库信息.xlsx'
    picking = DAP(distanceMatrixPath, warehouseInformation)  # 初始化
    _, l, path = picking.formingPath()
    print(f'最短路径为:{l}')
    print(f'最短路径对应的配置方式为:{path}')
    print(picking.result)
    input('Press <Enter>')
