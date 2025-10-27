import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tensorflow.keras.layers import Layer, Dense

# 设置随机种子以确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

class SingleHiddenLayerNetwork(Layer):
    def __init__(self, hidden_dim):
        super(SingleHiddenLayerNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = Dense(hidden_dim, activation='tanh')

    def call(self, inputs):
        return self.linear(inputs)

class PINNFeedForward(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PINNFeedForward, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 单隐藏层结构
        self.hidden_layer = Dense(hidden_dim, activation='tanh')
        self.output_layer = Dense(output_dim)

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        return self.output_layer(x)

#设置神经网络的结构和激活函数
###################################################################################################
# 使用Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# 定义PIDE的参数
mu = 0.1; theta = 0.3; T = 1; N = 60; h = T/N; d = 20; lam = 0.3

# 预分配内存和预计算常量
num_points = 1000
Me = 1
T_const = tf.constant(np.ones((num_points, 1)), dtype=tf.float32) * T
I = tf.constant(np.ones((num_points, 1)), dtype=tf.float32)

# 预计算一些常量
sqrt_h = math.sqrt(h)

# 创建模型
num_models = N
models_Y = [PINNFeedForward(input_dim=d, hidden_dim=d+10, output_dim=1) for _ in range(num_models)]
models_U = [PINNFeedForward(input_dim=d+1, hidden_dim=d+10, output_dim=1) for _ in range(num_models)]
###########################################################################################################

# 预分配内存用于存储变量, x0=1
x_init = [tf.ones((num_points, 1), dtype=tf.float32)  for _ in range(d)]

# 批量生成随机数
def pregenerate_random_variables(N, num_points, d, h, lam):
    # 预生成所有时间步,所有样本路径对应的布朗运动增量
    all_w = np.random.normal(0, 1, (N, num_points, d)) * sqrt_h

    # 预生成所有时间步,所有样本路径对应的跳跃次数
    all_Nn = np.random.poisson(lam * h, (N, num_points, 1))

    # 预生成所有时间步,所有样本路径上的的跳跃Zn并求和得到Jn
    max_Nn = int(np.max(all_Nn))
    all_Zn = np.zeros((N, num_points, max_Nn))
    all_Jn = np.zeros((N, num_points, 1))

    for n in range(N):
        for i in range(num_points):
            num_jumps = int(all_Nn[n, i, 0])
            if num_jumps > 0:
                Zn_vals = np.ones((num_jumps,)) * mu # 跳跃幅度是常数
                all_Zn[n, i, :num_jumps] = Zn_vals
                all_Jn[n, i, 0] = np.sum(Zn_vals)

    # 转换为TensorFlow常量
    all_w = tf.constant(all_w, dtype=tf.float32)
    all_Nn = tf.constant(all_Nn, dtype=tf.float32)
    all_Zn = tf.constant(all_Zn, dtype=tf.float32)
    all_Jn = tf.constant(all_Jn, dtype=tf.float32)

    return all_w, all_Nn, all_Zn, all_Jn, max_Nn

# 预生成积分点(计算退化分布)
e_points = tf.constant([0.1], dtype=tf.float32)
e_points = tf.reshape(e_points, (Me, 1))  # 形状为 [Me, 1]

#退化分布的密度
rho_points = tf.reshape(tf.constant([1], dtype=tf.float32), [1,1])
#######################################################################################################
# 定义单步训练函数
def train_step(n, x, x_next, w, Zn, Jn, max_num, t_val):
    with tf.GradientTape(persistent=True) as tape:
        # 监控变量
        variables = x.copy()
        for var in variables:
            tape.watch(var)

        # 准备当前时间节点上的输入
        inputs = tf.concat(x, axis=1)
        y = models_Y[n](inputs)

        # 计算解关于x的梯度
        gradients = [tape.gradient(y, var) for var in variables]
        SZ = tf.add_n(gradients)

        # 准备下一时间节点的输入
        inputs_next = tf.concat(x_next, axis=1)

        if n < N-1:
            y_next = models_Y[n+1](inputs_next)
        else:
            y_next = sum(x_i ** 2 for x_i in x_next) / d   # 精确解 u=(x1^2+x2^2+...+xd^2)/d

        # 计算鞅（跳跃部分）
        num_jumps = int(max_num)
        if num_jumps > 0:
            Zn_reshaped = tf.reshape(Zn[:, :num_jumps], (num_points, num_jumps, 1))
            inputs_jump = tf.tile(tf.expand_dims(inputs, 1), [1, num_jumps, 1])
            inputs_jump = tf.concat([inputs_jump, Zn_reshaped], axis=-1)
            inputs_jump_flat = tf.reshape(inputs_jump, [-1, d+1])
            U_vals = models_U[n](inputs_jump_flat)
            U_vals = tf.reshape(U_vals, (num_points, num_jumps)) # [num_points, num_jumps]
            Mar = tf.reduce_sum(U_vals, axis=1, keepdims=True)   # [num_points, 1]
        else:
            Mar = tf.zeros((num_points, 1), dtype=tf.float32)  # [num_points, 1]

        # 计算积分项intU(补偿项)
        # 创建积分点输入
        inputs_expanded = tf.tile(tf.expand_dims(inputs, 1), [1, Me, 1])  # [num_points, Me, d]
        e_points_expanded = tf.tile(tf.expand_dims(e_points, 0), [num_points, 1, 1])  # [num_points, Me, 1]
        inputs_g = tf.concat([inputs_expanded, e_points_expanded], axis=-1)  # [num_points, Me, d+1]

        inputs_g_flat = tf.reshape(inputs_g, [-1, d+1])
        intU_vals = models_U[n](inputs_g_flat)
        rho_points_expanded = tf.tile(rho_points, [num_points, 1])
        intU_vals = tf.reshape(intU_vals, (num_points, Me)) * rho_points_expanded # [num_points, Me]

        # 修正得到鞅项：Mar = ∑U(x, z) - λ * h * intU
        Mar_corrected = Mar - lam * h * intU_vals

        # 计算自由项f
        f = - (theta ** 2 + mu ** 2 * lam) * I

        # 计算BSDE的残差
        grad_w_sum = tf.constant(0.0, dtype=tf.float32)

        for i in range(d):
            grad_w_sum += gradients[i] * theta * w[:, i:i+1]

        pde_loss = y - h * f + grad_w_sum + Mar_corrected - y_next

        #将损失函数定义为残差平方的期望
        loss_value = tf.reduce_mean(tf.square(pde_loss))

    # 应用随机梯度下降更新一次神经网络Y, U的参数
    grads1 = tape.gradient(loss_value, models_Y[n].trainable_variables)
    grads1 = [tf.clip_by_value(grad, -1, 1) for grad in grads1]
    optimizer.apply_gradients(zip(grads1, models_Y[n].trainable_variables))

    grads2 = tape.gradient(loss_value, models_U[n].trainable_variables)
    grads2 = [tf.clip_by_value(grad, -1, 1) for grad in grads2]
    optimizer.apply_gradients(zip(grads2, models_U[n].trainable_variables))

    del tape
    return loss_value

#######################################################################################################
#参数训练过程
start_time = time.time()
loss_history = []
relative_L1_error_history = []

epochs = 1
for epoch in range(epochs):

    # 反向训练过程
    for n in range(N-1, -1, -1):

        dummy_tensor_1 = tf.ones((1, d), dtype=tf.float32)
        dummy_tensor_2 = tf.ones((1, d+1), dtype=tf.float32)
        output_tensor1 = models_Y[n](dummy_tensor_1)
        output_tensor2 = models_U[n](dummy_tensor_2)

        if n < N-1:
            Iteration = 50
            models_Y[n].set_weights(models_Y[n+1].get_weights())
            models_U[n].set_weights(models_U[n+1].get_weights())
            optimizer.learning_rate.assign(0.0005)
        else:
            Iteration = 12000
            optimizer.learning_rate.assign(0.0005)

        t_val = tf.constant(np.ones((num_points, 1)) * (n * h), dtype=tf.float32)

        for iteration in range(Iteration):

            if n < N-1:
                # 每隔100轮学习率减半
                if iteration > 0 and iteration % 150 == 0:
                    current_lr = optimizer.learning_rate.numpy()
                    new_lr = current_lr * 0.5
                    optimizer.learning_rate.assign(new_lr)
            else:
                # 每隔1000轮学习率减半
                if iteration > 0 and iteration % 3000 == 0:
                    current_lr = optimizer.learning_rate.numpy()
                    new_lr = current_lr * 0.5
                    optimizer.learning_rate.assign(new_lr)

            # 预生成所有随机数
            all_w, all_Nn, all_Zn, all_Jn, max_Nn = pregenerate_random_variables(n+1, num_points, d, h, lam)

            # 前向生成Xt的样本路径
            X = [x_init]
            for k in range(n+1):
                x = X[-1]
                Jn_current = all_Jn[k]

                x_next = []
                for i in range(d):
                   x_next.append(x[i] + theta * all_w[k, :, i:i+1] + Jn_current - lam * mu * h * I)

                X.append(x_next)

            #准备单步训练函数的输入
            x = X[n]
            x_next = X[n+1]
            w = all_w[n]
            Zn = all_Zn[n]
            Jn_current = all_Jn[n]

            #调用单步训练函数更新参数
            loss_value = train_step(n, x, x_next, w, Zn, Jn_current, max_Nn, t_val)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds")

#########################################################################################################
#保存和加载已训练完成的神经网络模型
batch1_dir = "saved_models"
os.makedirs(batch1_dir, exist_ok=True)

# 保存第一批模型
saved_models = {
    'Y': models_Y,
    'U': models_U,
}

for group_name, model_list in saved_models.items():
    for i, model in enumerate(model_list):
        model_path = os.path.join(batch1_dir, f"model_{group_name}_{i}")
        model.save(model_path)
        print(f"Saved dim20_240 model_{group_name}_{i} to {model_path}")

# 加载所有 Y 模型
models_Y = []
for i in range(num_models):
    model_path = os.path.join(batch1_dir, f"model_Y_{i}")
    loaded_model = tf.keras.models.load_model(model_path)
    models_Y.append(loaded_model)
    print(f"Loaded model_Y_{i}")

# 加载所有 U 模型
models_U = []
for i in range(num_models):
    model_path = os.path.join(batch1_dir, f"model_U_{i}")
    loaded_model = tf.keras.models.load_model(model_path)
    models_U.append(loaded_model)
    print(f"Loaded model_U_{i}")

#######################################################################################################

def simulate_Y_and_compute_error(T=1, theta=0.3, N=60, d=20, num_simulations=1):

    t = np.linspace(0, T, N+1)
    h = T / N

    for sim in range(num_simulations):
        # 初始化
        X_true = np.ones((N+1, d))
        Y_pred = np.zeros(N+1)
        Y_true = np.zeros(N+1)

        for i in range(N+1):
            # 计算真实值
            Y_true[i] = np.sum(X_true[i] ** 2) / d

            # 计算预测值
            x_input = tf.constant(X_true[i].reshape(1, d), dtype=tf.float32)
            Y_pred[i] = models_Y[min(i, N-1)](x_input).numpy()

            if i < N:
                # 生成随机跳跃
                num_jumps = np.random.poisson(lam * h)
                J = 0
                if num_jumps > 0:
                    Z_vals = np.ones((num_jumps,)) * mu
                    J = np.sum(Z_vals)

                # 更新X
                w_val = np.sqrt(h) * np.random.normal(0, 1, d)

                X_true[i+1] = X_true[i] + theta * w_val + J - lam * mu * h

    return t, X_true, Y_pred, Y_true

#####################################################################################################
# 绘图函数,绘制Yt的近似图
def plot_results_Yt(num_simulations=1):
    plt.figure(figsize=(12, 8))

    for i in range(num_simulations):
        t, X_true, Y_pred, Y_true = simulate_Y_and_compute_error(T=1, theta=0.3, N=60, d=20, num_simulations=1)

        if i == 0:
            plt.plot(t, Y_pred, 'r--', alpha=0.7, label='Predicted Y')
            plt.plot(t, Y_true, 'b-', alpha=0.7, label='True Y')
        else:
            plt.plot(t, Y_pred, 'r--', alpha=0.7)
            plt.plot(t, Y_true, 'b-', alpha=0.7)

    plt.title('Yt Process Comparison')
    plt.xlabel('Time')
    plt.ylabel('Yt')
    plt.grid(True)
    plt.legend()
    plt.show()

# 执行绘图
plot_results_Yt(num_simulations=10)