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

        self.input_layer = Dense(hidden_dim, activation='tanh')
        self.hidden_layer = SingleHiddenLayerNetwork(hidden_dim)
        self.output_layer = Dense(output_dim)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        return self.output_layer(x)

#设置神经网络的结构和激活函数
###################################################################################################

# 使用Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# 定义PIDE的参数
r = 1; theta = 1; T = 1; N = 30; h = T/N; d = 1; lam = 1

# Levy测度的参数
mu = 0.4; sigma = 0.25 #正态分布 N(mu,sigma^2)
delta = 0.7            #均匀分布 Uniform(-delta,delta)
lam_0 = 3              #指数分布 Exp(1/lam_0)

# 预分配内存和预计算常量
num_points = 1000
Me = 20   # Me=40对正态分布， Me=20对均匀、指数分布，  Me = 2 对伯努利分布
T_const = tf.constant(np.ones((num_points, 1)), dtype=tf.float32) * T
I = tf.constant(np.ones((num_points, 1)), dtype=tf.float32)

# 预计算一些常量
sqrt_h = math.sqrt(h)
two_delta = 2 * delta
delta_Me = two_delta / Me # 对均匀，正态分布
#delta_Me = delta / Me     # 对指数分布

# 创建模型
num_models = N
models_Y = [PINNFeedForward(input_dim=d, hidden_dim=d+20, output_dim=1) for _ in range(num_models)]    # 近似 Yti=u(ti,Xt)的神经网络族
models_U = [PINNFeedForward(input_dim=d+1, hidden_dim=d+20, output_dim=1) for _ in range(num_models)]  # 近似 Uti=u(ti,Xti+beta)-u(t,Xti)的神经网络族

###########################################################################################################
# 预分配内存用于存储变量, 取x0=pi/2
x_init = [tf.ones((num_points, 1), dtype=tf.float32) * math.pi/2 for _ in range(d)]

# 定义批量生成随机数的函数
def pregenerate_random_variables(N, num_points, d, h, lam, delta):
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
                #Zn_vals = np.random.normal(mu, sigma, num_jumps)       #e~N(mu,sigma^2)
                #Zn_vals = np.random.exponential(1/lam_0, num_jumps)    #e~Exp(1/lam_0)
                Zn_vals = np.random.uniform(-delta, delta, num_jumps)   #e~Uniform(-delta,delta)

                #single_bernoulli = np.random.binomial(1, 0.5, num_jumps)   #e~Bernoulli(p1=p2=0.5, -0.4, 0.8)
                #Zn_vals = single_bernoulli * (-0.4) + (np.ones((num_jumps,)) - single_bernoulli) * 0.8

                all_Zn[n, i, :num_jumps] = Zn_vals
                all_Jn[n, i, 0] = np.sum(Zn_vals)

    # 转换为TensorFlow常量
    all_w = tf.constant(all_w, dtype=tf.float32)
    all_Nn = tf.constant(all_Nn, dtype=tf.float32)
    all_Zn = tf.constant(all_Zn, dtype=tf.float32)
    all_Jn = tf.constant(all_Jn, dtype=tf.float32)

    return all_w, all_Nn, all_Zn, all_Jn, max_Nn

# 预生成积分点(计算伯努利分布)
#e_points = tf.constant([-0.4, 0.8], dtype=tf.float32)
#e_points = tf.reshape(e_points, (Me, 1))  # 形状为 [2, 1]

# 预生成积分点(计算指数分布)
#e_points = tf.constant([0 + m * delta_Me for m in range(Me)], dtype=tf.float32)
#e_points = tf.reshape(e_points, (Me, 1))  # 形状为 [20, 1]

# 预生成积分点(计算均匀分布)
e_points = tf.constant([-delta + m * delta_Me for m in range(Me)], dtype=tf.float32)
e_points = tf.reshape(e_points, (Me, 1))  # 形状为 [20, 1]

# 预生成积分点(计算正态分布)
#e_points = tf.constant([(mu-delta) + m * delta_Me for m in range(Me)], dtype=tf.float32)
#e_points = tf.reshape(e_points, (Me, 1))  # 形状为 [40, 1]

#levy测度密度函数
#rho_points = tf.reshape(1 / (math.sqrt(2 * math.pi) * sigma) * tf.exp(-(e_points - mu) ** 2 / (2 * sigma ** 2)), [1, Me])   #正态分布N(0,sigma)
rho_points = 1 / (2 * delta) * tf.constant(np.ones((1, Me)), dtype=tf.float32)   #均匀分布Uniform(-delta,delta)
#rho_points = tf.reshape(lam_0 * tf.exp(-lam_0 * e_points), [1, Me])   #指数分布exp(lam_0)
#rho_points = tf.reshape(tf.constant([0.5, 0.5], dtype=tf.float32), [1,2])   #伯努利分布
#######################################################################################################
# 定义单步训练函数
def train_step(n, x, x_next, w, Zn, Jn, max_num, t_val):
    with tf.GradientTape(persistent=True) as tape:
        # 监控变量
        variables = x.copy()
        for var in variables:
            tape.watch(var)

        # 准备输入当前时间节点上的输入
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
            y_next = sum(tf.sin(x_i) for x_i in x_next)

        # 计算鞅（跳跃部分）
        num_jumps = int(max_num)
        if num_jumps > 0:
            Zn_reshaped = tf.reshape(Zn[:, :num_jumps], (num_points, num_jumps, 1))
            inputs_jump = tf.tile(tf.expand_dims(inputs, 1), [1, num_jumps, 1])
            inputs_jump = tf.concat([inputs_jump, Zn_reshaped], axis=-1)
            inputs_jump_flat = tf.reshape(inputs_jump, [-1, d+1])
            U_vals = models_U[n](inputs_jump_flat)
            U_vals = tf.reshape(U_vals, (num_points, num_jumps))
            Mar = tf.reduce_sum(U_vals, axis=1, keepdims=True)
        else:
            Mar = tf.zeros((num_points, 1), dtype=tf.float32)

        # 计算积分项intU（补偿项）- 使用梯形公式
        # 创建积分点输入
        inputs_expanded = tf.tile(tf.expand_dims(inputs, 1), [1, Me, 1])  # [num_points, Me, d]
        e_points_expanded = tf.tile(tf.expand_dims(e_points, 0), [num_points, 1, 1])  # [num_points, Me, 1]
        inputs_g = tf.concat([inputs_expanded, e_points_expanded], axis=-1)  # [num_points, Me, d+1]

        inputs_g_flat = tf.reshape(inputs_g, [-1, d+1])
        intU_vals = models_U[n](inputs_g_flat)
        rho_points_expanded = tf.tile(rho_points, [num_points, 1])
        intU_vals = tf.reshape(intU_vals, (num_points, Me)) * rho_points_expanded # [num_points, Me]

        # 使用梯形公式近似积分
        # 梯形公式: ∫f(x)dx ≈ (Δx/2)[f(x0) + 2f(x1) + 2f(x2) + ... + 2f(x_{n-1}) + f(xn)]
        weights = tf.ones((Me,), dtype=tf.float32)
        weights = tf.tensor_scatter_nd_update(weights, [[0], [Me-1]], [0.5, 0.5])  # 首尾权重为0.5，中间为1

        # 应用梯形公式权重
        intU_weighted = intU_vals * weights  # [num_points, Me]
        intU = tf.reduce_sum(intU_weighted, axis=1, keepdims=True) * delta_Me  # [num_points, 1], 对于均匀，正态，指数分布

        # [num_points, 1], 对于伯努利分布
        #intU = tf.reduce_sum(intU_vals, axis=1, keepdims=True)

        # 修正得到鞅项：Mar = ∑U(x, z) - λ * h * intU
        Mar_corrected = Mar - lam * h * intU

        # 计算自由项f
        f = - r * y * tf.exp(SZ) / tf.exp((tf.exp(-r * (T_const - t_val)) * sum(tf.cos(x_i) for x_i in x))) \
            + 0.5 * theta ** 2 * (1/d) * y \
            - lam * tf.exp(-r * (T_const - t_val)) * sum((tf.sin(x_i) * math.sin(delta) / delta - tf.sin(x_i)) for x_i in x) #均匀分布非局部积分项
            #- lam * tf.exp(-r * (T_const - t_val)) * sum((0.5 * tf.sin(x_i - 0.4 * I) + 0.5 * tf.sin(x_i + 0.8 * I) - tf.sin(x_i)) for x_i in x) + lam * (0.2) * SZ   #伯努利分布的非局部积分项
            #- lam * tf.exp(-r * (T_const - t_val)) * sum(((lam_0 / (lam_0**2 + 1)) * tf.cos(x_i)  - (1/(lam_0**2 + 1)) * tf.sin(x_i)) for x_i in x) + lam * (1/lam_0) * SZ   #指数分布非局部积分项
            #- lam * tf.exp(-r * (T_const - t_val)) * sum((tf.sin(x_i) * math.exp(-sigma ** 2) - tf.sin(x_i)) for x_i in x) + lam * mu * SZ #正态分布非局部积分项

        # 计算BSDEJ的残差
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
# 参数训练过程
start_time = time.time()
loss_history = []
relative_L1_error_history = []

epochs = 1
for epoch in range(epochs):

    # 反向训练过程, n = N-1,N-2,...,0
    for n in range(N-1, -1, -1):

        # 预先调用神经网络，以获得可训练的参数
        dummy_tensor_1 = tf.ones((1, d), dtype=tf.float32)
        dummy_tensor_2 = tf.ones((1, d+1), dtype=tf.float32)
        output_tensor1 = models_Y[n](dummy_tensor_1)
        output_tensor2 = models_U[n](dummy_tensor_2)


        if n < N-1:
            models_Y[n].set_weights(models_Y[n+1].get_weights())
            models_U[n].set_weights(models_U[n+1].get_weights())
            # 节点tn,n<N-1时，将后一个节点tn+1对应的神经网络已优化到最优的参数theta*_n+1 赋给 tn 对应的网络参数theta_n

            Iteration = 300
            #相邻两个时间节点对应的神经网络，最优参数非常接近，被赋予了后一个节点的最优网络参数后，只需再作少量次数更新

            optimizer.learning_rate.assign(0.0005)
        else:
            Iteration = 12000
            #第N-1个时间节点对应的网络，需从最初始的参数开始训练，需进行大量次数的参数更新以确保最优

            optimizer.learning_rate.assign(0.0005)

        t_val = tf.constant(np.ones((num_points, 1)) * (n * h), dtype=tf.float32) # tn=n*h

        for iteration in range(Iteration):

            if n < N-1:
                # 训练节点tn,n<N-1 对应的网络时， 每隔150轮学习率减半
                if iteration > 0 and iteration % 150 == 0:
                    current_lr = optimizer.learning_rate.numpy()
                    new_lr = current_lr * 0.5
                    optimizer.learning_rate.assign(new_lr)
            else:
                # 训练第N-1节点对应的网络时，每隔3000轮学习率减半
                if iteration > 0 and iteration % 3000 == 0:
                    current_lr = optimizer.learning_rate.numpy()
                    new_lr = current_lr * 0.5
                    optimizer.learning_rate.assign(new_lr)

            # 预生成所有随机数
            all_w, all_Nn, all_Zn, all_Jn, max_Nn = pregenerate_random_variables(n+1, num_points, d, h, lam, delta)

            # 前向生成Xt的样本路径
            X = [x_init]
            for k in range(n+1):
                x = X[-1]
                Jn_current = all_Jn[k]

                x_next = []
                for i in range(d):
                    x_next.append(x[i] + theta * all_w[k, :, i:i+1] + Jn_current - lam * (0) * I * h)
                X.append(x_next)

            #准备单步训练函数的输入
            x = X[n]
            x_next = X[n+1]
            w = all_w[n]
            Zn = all_Zn[n]
            Jn_current = all_Jn[n]

            #调用单步训练函数，计算损失函数的值并更新一次参数
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
        print(f"Saved Uniform model_{group_name}_{i} to {model_path}")

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
# 误差可视化,绘图

#训练完成后，模拟样本路径，以绘制Yt的近似图
def simulate_Y(T=1, theta=1, delta=0.7, N=30, d=1, num_simulations=1):
    t = np.linspace(0, T, N+1)
    h = T / N

    # 预分配内存
    X_true = np.ones((N+1, d)) * math.pi/2
    Y_pred = np.zeros(N+1)
    Y_true = np.zeros(N+1)
    jump_times = []  # 记录跳跃发生的时间点

    for i in range(N+1):
        # 计算真实值
        Y_true[i] = np.exp(-r * (T - i * h)) * np.sum(np.sin(X_true[i]))

        # 计算预测值
        x_input = tf.constant(X_true[i].reshape(1, d), dtype=tf.float32)
        Y_pred[i] = models_Y[min(i, N-1)](x_input).numpy()

        if i < N:
            # 生成随机跳跃
            num_jumps = np.random.poisson(lam * h)
            J = 0
            if num_jumps > 0:
                Z_vals = np.random.uniform(-delta, delta, num_jumps) # Uniform
                #Z_vals = np.random.normal(mu, sigma, num_jumps)     # Normal
                #Z_vals = np.random.exponential(1/lam_0, num_jumps)  # Exp

                #ber = np.random.binomial(1, 0.5, num_jumps)         # Bernoulli
                #Z_vals = ber * (-0.5) + (np.ones((num_jumps,)) - ber) * 1

                J = np.sum(Z_vals)
                # 记录跳跃发生的时间段 [t_i, t_{i+1})
                jump_times.append((t[i], t[i+1]))

            # 更新X
            w_val = np.sqrt(h) * np.random.normal(0, 1, d)
            X_true[i+1] = X_true[i] + theta/math.sqrt(d) * w_val + J - (0) * lam * h

    return t, X_true, Y_pred, Y_true, jump_times

#绘制u(t,x)，Du(t,x)的近似图
def simulate_u(T=1, n=0, a=0.6, b=1.4, Nx=40, d=1, num_simulations=1):

    x = np.linspace(a, b, Nx+1)
    u_exact = np.zeros(Nx+1)
    u_pred = np.zeros(Nx+1)
    partial_x_u_exact = np.zeros(Nx+1)
    partial_x_u_pred = np.zeros(Nx+1)

    for i in range(Nx+1):
        u_exact[i] = math.exp(-r * (T - n * h)) * math.sin(x[i])
        partial_x_u_exact[i] = math.exp(-r * (T - n * h)) * math.cos(x[i])
        x_test = tf.constant(x[i].reshape(1, d), dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_test)
            u_pred_tf = models_Y[n](x_test)
            u_pred[i] = u_pred_tf.numpy()
            partial_x_u_pred[i] = tape.gradient(u_pred_tf, x_test).numpy()

        del tape

    return x, u_exact, u_pred, partial_x_u_exact, partial_x_u_pred

#####################################################################################################
# 绘制Yt的近似路径图
def plot_results_Yt_alternative(num_simulations=1):
    plt.figure(figsize=(12, 8))

    # 创建图例标签标志
    jump_label_added_pred = False
    jump_label_added_true = False

    for i in range(num_simulations):
        t, X_true, Y_pred, Y_true, jump_times = simulate_Y(T=1, theta=1, delta=0.7, N=30, d=1, num_simulations=1)

        # 绘制基本线条 - 使用更深的颜色
        if i == 0:
            base_line_pred, = plt.plot(t, Y_pred, 'darkred', linestyle='--', alpha=0.7, linewidth=1.5, label='Predicted Y')
            base_line_true, = plt.plot(t, Y_true, 'darkblue', alpha=0.7, linewidth=1.5, label='True Y')
        else:
            base_line_pred, = plt.plot(t, Y_pred, 'darkred', linestyle='--', alpha=0.7, linewidth=1.5)
            base_line_true, = plt.plot(t, Y_true, 'darkblue', alpha=0.7, linewidth=1.5)

        # 在跳跃时间段绘制不同颜色的粗线条
        for j, (jump_start, jump_end) in enumerate(jump_times):
            jump_indices = np.where((t >= jump_start) & (t <= jump_end))[0]

            if len(jump_indices) > 0:
                # 只在第一次需要时添加图例标签
                pred_label = 'Predicted Y path exhibits jumps' if not jump_label_added_pred else ""
                true_label = 'True Y path exhibits jumps' if not jump_label_added_true else ""

                # 使用鲜艳的颜色突出跳跃区间
                plt.plot(t[jump_indices], Y_pred[jump_indices], 'orange',
                        alpha=0.9, linewidth=3, label=pred_label)
                plt.plot(t[jump_indices], Y_true[jump_indices], 'lime',
                        alpha=0.9, linewidth=3, label=true_label)

                # 标记已添加图例标签
                if pred_label:
                    jump_label_added_pred = True
                if true_label:
                    jump_label_added_true = True

    plt.title('Uniform distribution')
    plt.xlabel('t')
    plt.ylabel('Yt')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# 执行修改后的替代方案绘图
plot_results_Yt_alternative(num_simulations=8)

# 绘制函数u(tn,x),Du(tn,x)的近似函数曲线
def plot_results_u(num_simulations=1):
    # 获取数据
    x0, u_exact_0, u_pred_0, partial_x_u_exact_0, partial_x_u_pred_0 \
        = simulate_u(T=1, n=0, a=0 , b=math.pi, Nx=40, d=1, num_simulations=1)

    x1, u_exact_1, u_pred_1, partial_x_u_exact_1, partial_x_u_pred_1 \
        = simulate_u(T=1, n=10, a=0, b=math.pi, Nx=40, d=1, num_simulations=1)

    x2, u_exact_2, u_pred_2, partial_x_u_exact_2, partial_x_u_pred_2 \
        = simulate_u(T=1, n=20, a=0, b=math.pi, Nx=40, d=1, num_simulations=1)

    x3, u_exact_3, u_pred_3, partial_x_u_exact_3, partial_x_u_pred_3 \
        = simulate_u(T=1, n=29, a=0, b=math.pi, Nx=40, d=1, num_simulations=1)

    # 创建图形
    plt.figure(figsize=(18, 12))  # 增加高度以更好地显示图例

    # 第一个子图  u(0,x), 调用神经网络Y[0]
    plt.subplot(2, 2, 1)
    l1, = plt.plot(x0, u_exact_0, 'b-', alpha=0.7, label='exact_u')
    l2, = plt.plot(x0, u_pred_0, 'r--', alpha=0.7, label='pred_u')
    #l1, = plt.plot(x0, partial_x_u_exact_0, 'b-', alpha=0.7, label='exact')
    #l2, = plt.plot(x0, partial_x_u_pred_0, 'r--', alpha=0.7, label='pred')
    plt.xlabel('x', fontsize=9)
    plt.ylabel('u(0,x)', fontsize=9)
    plt.legend(loc='best', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # 第二个子图  u(0.33,x), 调用Y[10]
    plt.subplot(2, 2, 2)
    l3, = plt.plot(x1, u_exact_1, 'b-', alpha=0.7, label='exact')
    l4, = plt.plot(x1, u_pred_1, 'r--', alpha=0.7, label='pred')
    #l3, = plt.plot(x1, partial_x_u_exact_1, 'b-', alpha=0.7, label='exact')
    #l4, = plt.plot(x1, partial_x_u_pred_1, 'r--', alpha=0.7, label='pred')
    plt.xlabel('x', fontsize=9)
    plt.ylabel('u(0.33,x)', fontsize=9)
    plt.legend(loc='best', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # 第三个子图  u(0.66,x), 调用Y[20]
    plt.subplot(2, 2, 3)
    l5, = plt.plot(x1, u_exact_2, 'b-', alpha=0.7, label='exact')
    l6, = plt.plot(x1, u_pred_2, 'r--', alpha=0.7, label='pred')
    #l5, = plt.plot(x2, partial_x_u_exact_2, 'b-', alpha=0.7, label='exact')
    #l6, = plt.plot(x2, partial_x_u_pred_2, 'r--', alpha=0.7, label='pred')
    plt.xlabel('x', fontsize=9)
    plt.ylabel('u(0.66,x)', fontsize=9)
    plt.legend(loc='best', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # 第四个子图  u(0.96,x),  调用Y[29]
    plt.subplot(2, 2, 4)
    l7, = plt.plot(x3, u_exact_3, 'b-', alpha=0.7, label='exact')
    l8, = plt.plot(x3, u_pred_3, 'r--', alpha=0.7, label='pred')
    #l7, = plt.plot(x3, partial_x_u_exact_3, 'b-', alpha=0.7, label='exact')
    #l8, = plt.plot(x3, partial_x_u_pred_3, 'r--', alpha=0.7, label='pred')
    plt.xlabel('x', fontsize=9)
    plt.ylabel('u(0.96,x)', fontsize=9)
    plt.legend(loc='best', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # 调整布局
    plt.tight_layout()

    # 保存和显示图像
    plt.savefig('img.png', dpi=300, bbox_inches='tight')  # 添加文件扩展名和提高质量
    plt.show()

# 执行绘图
plot_results_u(num_simulations=1)