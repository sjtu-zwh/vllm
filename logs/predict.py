import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def process_csv_and_fit(csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 提取前四列作为ranks
    rank_columns = df.columns[1:5]
    
    # 对前四个数相同的行分组，并计算computing_time的平均值
    grouped = df.groupby(rank_columns.tolist())['Computing Time (ms)'].mean().reset_index()
    grouped['Total Tokens'] = df.groupby(rank_columns.tolist())['Total Tokens'].first().values
    
    # 计算每行的最大非零rank
    max_nonzero_ranks = []
    for _, row in grouped.iterrows():
        nonzero_ranks = [int(col) for col, val in zip(rank_columns, row[rank_columns]) if val != 0]
        max_nonzero_rank = max(nonzero_ranks) if nonzero_ranks else 0
        max_nonzero_ranks.append(max_nonzero_rank)
    
    # 计算X = max_nonzero_rank * total_tokens
    grouped['max_nonzero_rank'] = max_nonzero_ranks
    grouped['X'] = grouped['max_nonzero_rank'] * grouped['Total Tokens']
    
    # 提取X和y
    X = grouped['X'].values.reshape(-1, 1)
    y = grouped['Computing Time (ms)'].values

    # 分割X和y为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 线性回归拟合
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 打印回归结果
    print("回归系数 (斜率):", model.coef_[0])
    print("截距:", model.intercept_)
    print("训练集 R^2 分数:", model.score(X_train, y_train))
    print("测试集 R^2 分数:", model.score(X_test, y_test))
    print("相对误差:", np.mean(np.abs((y_test - model.predict(X_test)) / y_test)) * 100, "%")
    
    # 绘制散点图和回归线
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.xlabel('X (max_nonzero_rank * total_tokens)')
    plt.ylabel('y (computing_time)')
    plt.title('Linear Regression Fit (Averaged Data)')
    plt.legend()
    plt.savefig('linear_regression_fit.png')
    plt.show()
    
    return grouped, model

# 示例用法
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    df, model = process_csv_and_fit(csv_path)
    
    # 打印处理后的数据
    print("\n处理后的数据:")
    print(df)