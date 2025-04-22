import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

def process_csv_and_fit_regression_1(csv_path):
    # --- 数据预处理 ---
    df = pd.read_csv(csv_path)
    
    # 解析rank值（从表头获取）
    rank_values = [int(col) for col in df.columns[1:5]]  # 提取4,8,16,32
    print(f"使用的rank值: {rank_values}")

    df = df[df['Total Tokens'] > 0].copy()
    print(f"排除零值后数据量: {len(df)}")
    
    # 计算每个rank列的特征：rank值 × 数量
    for i, rank in enumerate(rank_values, start=1):
        col_name = df.columns[i]  # 对应数量列（如第2列是"4"的数量）
        df[f'feature_{i}'] = df[col_name] * rank
    
    # 分组计算平均计算时间（保留所有特征列）
    group_cols = df.columns[1:5].tolist()
    features = [f'feature_{i}' for i in range(1, 5)]
    grouped = df.groupby(group_cols).agg({
        'Computing Time (ms)': 'mean',
        'Total Tokens': 'first',
        **{f: 'first' for f in features}  # 保留所有特征列
    }).reset_index()
    
    # 提取4个新特征和标签
    features = [f'feature_{i}' for i in range(1, 5)]
    X = grouped[features].values
    y = grouped['Computing Time (ms)'].values
    
    # --- 训练回归模型 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # --- 输出结果 ---
    print("\n回归结果:")
    print("特征含义:")
    for i, rank in enumerate(rank_values, start=1):
        print(f"  feature_{i}: (num x {rank})")
    
    print("\n回归系数:")
    for i, coef in enumerate(model.coef_, start=1):
        print(f"  feature_{i}: {coef:.4f}")
    print(f"截距: {model.intercept_:.4f}")
    
    print("\n模型性能:")
    print(f"训练集 R²: {model.score(X_train, y_train):.4f}")
    print(f"测试集 R²: {model.score(X_test, y_test):.4f}")
    print(f"相对误差: {np.mean(np.abs((y_test - model.predict(X_test)) / y_test))*100:.4f}%")
    
    # --- 可视化所有特征的边际效应 ---
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(X[:, i], y, alpha=0.5)
        
        # 绘制单变量回归线
        x_vals = np.linspace(X[:, i].min(), X[:, i].max(), 100)
        y_vals = model.intercept_ + model.coef_[i] * x_vals
        plt.plot(x_vals, y_vals, 'r--')
        
        plt.xlabel(f'feature_{i+1} (num x {rank_values[i]})')
        plt.ylabel('Computing Time (ms)')
        plt.title(f'Rank {rank_values[i]} ')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rank_features_regression.png')
    plt.show()
    
    return model, grouped

def process_csv_and_fit_regression_2(csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 提取前四列作为ranks
    rank_columns = df.columns[1:5]
    
    # 对前四个数相同的行分组，并计算computing_time的平均值
    grouped = df.groupby(rank_columns.tolist())['Computing Time (ms)'].mean().reset_index()
    grouped['Total Tokens'] = df.groupby(rank_columns.tolist())['Total Tokens'].first().values
    
    # # 计算每行的最大非零rank
    # max_nonzero_ranks = []
    # for _, row in grouped.iterrows():
    #     nonzero_ranks = [int(col) for col, val in zip(rank_columns, row[rank_columns]) if val != 0]
    #     max_nonzero_rank = max(nonzero_ranks) if nonzero_ranks else 0
    #     max_nonzero_ranks.append(max_nonzero_rank)
    
    # # 计算X = max_nonzero_rank * total_tokens
    # grouped['max_nonzero_rank'] = max_nonzero_ranks
    # grouped['X'] = grouped['max_nonzero_rank'] * grouped['Total Tokens']

    grouped['X'] = 32 * grouped['Total Tokens']
    
    # 提取X和y
    X = grouped['X'].values.reshape(-1, 1)
    y = grouped['Computing Time (ms)'].values

    # 分割X和y为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 线性回归拟合
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 打印回归结果
    print("回归系数 (斜率):", model.coef_[0])
    print("截距:", model.intercept_)
    print("训练集 R^2 分数:", model.score(X_train, y_train))
    print("测试集 R^2 分数:", model.score(X_test, y_test))
    
    # 绘制散点图和回归线
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.xlabel('X (max_nonzero_rank * total_tokens)')
    plt.ylabel('y (computing_time)')
    plt.title('Linear Regression Fit (Averaged Data)')
    plt.legend()
    plt.savefig('linear_regression_fit.png')
    plt.show()
    
    return model, grouped

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model, df = process_csv_and_fit_regression_1(csv_path)
    
    # # 打印处理后的数据（含新特征）
    # print("\n处理后的数据（前5行）:")
    # print(df.head())