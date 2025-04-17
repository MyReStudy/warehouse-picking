# 仓库拣货路径规划算法

:wave:欢迎使用仓库拣货路径规划问题求解算法项目。在本项目中，点击 “single_block” 与 “multi_block”，即可运行可执行文件，直观感受算法的求解过程。由于论文尚未正式发表，出于学术规范与保密性考虑，部分核心代码暂未公开，现阶段您仍可进行运行操作，探索算法在实际应用中的表现。

# 文件说明
:smiley_cat:**single_block**：单块仓拣货路径规划算法。

--test：一组测试数据

--DAP.py：程序入口，运行后得到规划结果如图
![image](https://github.com/LiuYuqier/warehouse-picking/blob/main/res/single_block_result.png)

:smiley_cat:**multi_block**：多块仓拣货路径规划算法。

--warehouse_data_4_5：训练好的模型，可以进行调用求解

--run.py：程序入口，运行后调用模型完成随机生成的订单路径规划求解，某次运行结果如图（不完整截图）：
![image](https://github.com/LiuYuqier/warehouse-picking/blob/main/res/multi_block_result.png)

该部分代码参考了https://github.com/wouterkool/attention-learn-to-route，相关论文：Attention, Learn to Solve Routing Problems!

# 联系我
:blush:若您对本项目感兴趣，欢迎通过Github 与我取得联系，也可直接发送邮件至 liuyuqi322@163.com，期待与您交流探讨，共同推进仓库拣货路径规划算法研究。

该项目相关论文链接：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10857038
。若您在后续研究或应用中使用到本项目涉及的思想与方法，请予以规范引用。
