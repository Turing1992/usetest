#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 11.05.py
# @Author: ruixi L
# @Date  : 2019/11/5

import tensorflow as tf
tf.set_random_seed(777)
datax=[[0,0],[0,1],[1,0],[1,1]]
datay=[[0],[1],[1],[1]]
# 3.初始化X，Y占位符（5分）
X=tf.placeholder(tf.float32,shape=[None,2],name='x')
Y=tf.placeholder(tf.float32,shape=[None,1],name='y')
# 4.初始化W1，b1张量（5分）
w1=tf.Variable(tf.random_normal([2,2]),dtype=tf.float32,name='w1')
b1=tf.Variable(tf.random_normal([2]),dtype=tf.float32,name='b1')
# 5.设置隐藏层模型，使用sigmoid函数（5分）
z1=tf.matmul(X,w1)+b1
a1=tf.sigmoid(z1)
# 6.初始化W2，b2张量（5分）
w2=tf.Variable(tf.random_normal([2,1]),dtype=tf.float32,name='w2')
b2=tf.Variable(tf.random_normal([1]),dtype=tf.float32,name='b2')
a2=tf.matmul(a1,w2)+b2
# 7.设置hypothesis预测模型（5分）
hypothesis=tf.sigmoid(a2,name='hypothesis')
# 8.设置代价函数（5分）
cost=-tf.reduce_mean(tf.multiply(Y,tf.log(hypothesis))+tf.multiply((1-Y),tf.log(1-hypothesis)))
# 9.使用梯度下降优化器查找最优解（5分）
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1)),dtype=tf.float32),name='accuracy')
# 10.创建会话（5分）
sess=tf.Session()
saver=tf.train.Saver()
# 11.初始化全局变量（5分）
sess.run(tf.global_variables_initializer())
# 12.迭代训练100次，每100次输出一次cost（5分）
for step in range(100):
    cost_val,_=sess.run([cost,optimizer],feed_dict={X:datax,Y:datay})
    result,acc=sess.run([hypothesis,accuracy],feed_dict={X:datax,Y:datay})
print('预测值',result,'准确度',acc)
saver.save(sess,'./checkpoint/mymodel',global_step=100)
# 13.输出预测值、准确度（5分）
# 14.按检查点方式保存网络结构和参数（5分）
# 15.新建一个python 程序，恢复网络结构和参数（10分，每点5分）
# 16.在新建程序中输出预测值、准确度（5分）
# 17.简述github的两大作用（10分，每点5分）
'''
    1.github上各国各企业的coder在上面存放开源代码，方便下载学习，免费插件众多，可以为自己的代码进行查错，规范化
    2.可以将自己的代码链接至自己的电脑，使用git bash窗口控制自己的代码上传至github服务器，也可以控制自己的github仓库文件
'''
# 18.使用git bash 同步一个文件到github服务器（5分）