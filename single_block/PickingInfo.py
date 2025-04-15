import pandas as pd


# class PickingState:
#     def __init__(self, id):
#         self.id = id
#         # self.decision = []
#         self.upFlag = 0
#         self.downFlag = 0
#         self.middleFlag = 0
#         self.routeNum = 0
#
#     def getId(self):
#         return self.id
#
#     def isOneRoute(self):
#         flag = self.upFlag + self.downFlag + self.middleFlag
#         if (flag == 3 or flag == 1):
#             self.routeNum = 1
#             return True
#         else:
#             self.routeNum = 2
#             return False
def getAisleList(warehouseInformation):  # 所有存着货物的列 按照从小到大顺序排列
    aislelist = sorted(warehouseInformation['aisle'].unique())
    return aislelist

def initResult(aisleNum):
    columns1 = [i for i in aisleNum]  # 所有非空拣货巷道（给阶段stage标号）
    index1 = ['state1', 'state2', 'state3', 'state4_1', 'state4_2']  # 不同的状态
    initResult = pd.DataFrame(columns=columns1, index=index1)
    return initResult

stateList = ['state1', 'state2', 'state3', 'state4_1', 'state4_2']

cofList=[1,2,3,4]

# 状态转移表
#下一个状态:{上一个状态:[中间巷道可能的配置]}
stateTransfer = {
    'state1':{
        'state1':[2,3,4],
        'state2':1,
        'state3':1,
        'state4_1':1,
        'state4_2':1
    },
    'state2':{
        'state1':1,
        'state2':2,
        'state4_1':[3,4]
    },
    'state3':{
        'state1':1,
        'state3':3,
        'state4_1':[2,4]
    },
    'state4_1':{
        'state1':1,
        'state4_1':[2,3,4]
    },
    'state4_2':{
        'state2':[3,4],
        'state3':[4,2],
        'state4_2':[2,3,4]
    }
}

# StateAndconfigs = {
#     'state1' : [1,2,3,4],
#     'state2' : [1,2,3,4],
#     'state3' : [1,2,3,4],
#     'state4_1' : [1,2,3,4],
#     'state4_2' : [2,3,4]
#
# }

initAndLastState = {  # 初始阶段与最后一个阶段不同的状态下可能的配置是唯一的
    'state1': 1,
    'state2': 2,
    'state3': 3,
    'state4_1': 4,
    'state4_2': 4
}

# def getOneRouteFlag(method,state,preflag):#左边和中间的配置、右边的状态、左边之前成一条路了不
#     #flag=0永远会是一条路了不用管了; flag=1目前是一条路可能会变成两条路; flag=2是两条路
#     preState=method[0]
#     vertical=method[1]
#     newState=state
#     nowflag=-1
#     if preflag == 1:
#         nowflag=1
#     elif preflag == 2:
#         if vertical==1:
#             nowflag=1
#         else:
#             nowflag=2
#     elif preflag == 0:
#         if vertical==1:
#             nowflag=1
#         elif newState=='state4':
#             nowflag=2
#     return nowflag

# def getInitFlag(state):
#     if state=='state1':
#         return 1
#     elif state=='state2' or state=='state3':
#         return 0
#     else:
#         return 2
#
# def getFinalFlag(preflag,vertical):
#     if preflag==1:
#         return 1
#     elif preflag==0:
#         return 1
#     else:
#         if vertical==1:
#             return 1
#         else:
#             return 2


if __name__ == '__main__':
    initResult(6)