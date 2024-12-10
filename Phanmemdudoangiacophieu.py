from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox

df = pd.read_csv('Stock.csv')
Dt_train, Dt_test = train_test_split(df, test_size=0.3, shuffle=False)
X_train = np.array(Dt_train[['Open','High','Low','Volume']].values) 
y_train = np.array(Dt_train['Close'])
X_test = np.array(Dt_test[['Open','High','Low','Volume']].values) 
y_test = np.array(Dt_test['Close'])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Ridge Regression
ridge_model = Ridge(alpha=0.0001)
ridge_model.fit(X_train, y_train)

# Neural Network (MLPRegressor)
mlp_model = MLPRegressor(hidden_layer_sizes=(200,500),solver='lbfgs',max_iter=100,activation='identity' )
mlp_model.fit(X_train, y_train)

# Stacking
base_models = [('linear', linear_model), ('ridge', ridge_model), ('mlp', mlp_model)]
stacking_model = StackingRegressor(estimators=base_models)
stacking_model.fit(X_train, y_train)

def evaluate_models(X_test, y_test):
    # Linear Regression
    linear_predictions = linear_model.predict(X_test)
    linear_r2 = r2_score(y_test, linear_predictions)
    linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
    linear_mae = mean_absolute_error(y_test, linear_predictions)
    linear_nse = str((1-(np.sum((y_test-linear_predictions)**2)/np.sum((y_test-np.mean(y_test))**2))))

    # Ridge Regression
    ridge_predictions = ridge_model.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_predictions)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
    ridge_mae = mean_absolute_error(y_test, ridge_predictions)
    ridge_nse = str((1-(np.sum((y_test-ridge_predictions)**2)/np.sum((y_test-np.mean(y_test))**2))))

    # MLPRegressor
    mlp_predictions = mlp_model.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_predictions)
    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_predictions))
    mlp_mae = mean_absolute_error(y_test, mlp_predictions)
    mlp_nse = str((1-(np.sum((y_test-mlp_predictions)**2)/np.sum((y_test-np.mean(y_test))**2))))

    # StackingRegressor
    stacking_predictions = stacking_model.predict(X_test)
    stacking_r2 = r2_score(y_test, stacking_predictions)
    stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_predictions))
    stacking_mae = mean_absolute_error(y_test, stacking_predictions)
    stacking_nse = str((1-(np.sum((y_test-stacking_predictions)**2)/np.sum((y_test-np.mean(y_test))**2))))

    return {
        'Linear Regression': {'R2': linear_r2, 'NSE': linear_nse, 'RMSE': linear_rmse, 'MAE': linear_mae},
        'Ridge Regression': {'R2': ridge_r2, 'NSE': ridge_nse, 'RMSE': ridge_rmse, 'MAE': ridge_mae},
        'MLPRegressor': {'R2': mlp_r2, 'NSE': mlp_nse, 'RMSE': mlp_rmse, 'MAE': mlp_mae},
        'StackingRegressor': {'R2': stacking_r2, 'NSE': stacking_nse, 'RMSE': stacking_rmse, 'MAE': stacking_mae}
    }

def predict_new_data_linear():
    try:
        new_data = [float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get())]
        metrics = evaluate_models(X_test, y_test)
        result_text = f"Giá trị dự đoán: {linear_model.predict([new_data])[0]}\n"
        for model_name, scores in metrics.items():
            result_text += f"{model_name} - R2: {scores['R2']}, NSE: {scores['NSE']}, RMSE: {scores['RMSE']}, MAE: {scores['MAE']}\n"

        result_label.config(text=result_text)
    except ValueError:
        messagebox.showerror("Lỗi", "Đầu vào không hợp lệ. Vui lòng nhập đúng dữ liệu số.")

def predict_new_data_ridge():
    try:
        new_data = [float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get())]
        metrics = evaluate_models(X_test, y_test)
        result_text = f"Giá trị dự đoán: {ridge_model.predict([new_data])[0]}\n"
        result_label.config(text=result_text)
        for model_name, scores in metrics.items():
            result_text += f"{model_name} - R2: {scores['R2']}, NSE: {scores['NSE']}, RMSE: {scores['RMSE']}, MAE: {scores['MAE']}\n"

        result_label.config(text=result_text)
    except ValueError:
        messagebox.showerror("Lỗi", "Đầu vào không hợp lệ. Vui lòng nhập đúng dữ liệu số.")

def predict_new_data_mlp():
    try:
        new_data = [float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get())]
        metrics = evaluate_models(X_test, y_test)
        result_text = f"Giá trị dự đoán: {mlp_model.predict([new_data])[0]}\n"
        result_label.config(text=result_text)
        for model_name, scores in metrics.items():
            result_text += f"{model_name} - R2: {scores['R2']}, NSE: {scores['NSE']}, RMSE: {scores['RMSE']}, MAE: {scores['MAE']}\n"
        result_label.config(text=result_text)
    except ValueError:
        messagebox.showerror("Lỗi", "Đầu vào không hợp lệ. Vui lòng nhập đúng dữ liệu số.")

def predict_new_data_stacking():
    try:
        new_data = [float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get())]
        metrics = evaluate_models(X_test, y_test)
        result_text = f"Giá trị dự đoán: {stacking_model.predict([new_data])[0]}\n"
        result_label.config(text=result_text)
        for model_name, scores in metrics.items():
            result_text += f"{model_name} - R2: {scores['R2']}, NSE: {scores['NSE']}, RMSE: {scores['RMSE']}, MAE: {scores['MAE']}\n"
        result_label.config(text=result_text)
    except ValueError:
        messagebox.showerror("Lỗi", "Đầu vào không hợp lệ. Vui lòng nhập đúng dữ liệu số.")
# Create main window
window = Tk()
window.title("Form Dự Đoán Giá Cổ Phiếu")
window.geometry("800x400")

# Create and pack widgets
header_label = Label(window, text="Nhập dữ liệu để dự đoán giá cổ phiếu:")
header_label.pack()
label1 = Label(window, text="Open:")
label1.pack()
entry1 = Entry(window)
entry1.pack()

label2 = Label(window, text="High:")
label2.pack()
entry2 = Entry(window)
entry2.pack()

label3 = Label(window, text="Low:")
label3.pack()
entry3 = Entry(window)
entry3.pack()

label4 = Label(window, text="Volume:")
label4.pack()
entry4 = Entry(window)
entry4.pack()

predict_button = Button(window, text="Giá trị dự đoán dựa trên mô hình LinearRegression:", command=predict_new_data_linear)
predict_button.pack()

predict_button = Button(window, text="Giá trị dự đoán dựa trên mô hình RidgeRegression:", command=predict_new_data_ridge)
predict_button.pack()

predict_button = Button(window, text="Giá trị dự đoán dựa trên mô hình MLPRegression:", command=predict_new_data_mlp)
predict_button.pack()

predict_button = Button(window, text="Giá trị dự đoán dựa trên mô hình StackingRegression:", command=predict_new_data_stacking)
predict_button.pack()

result_label = Label(window, text="")
result_label.pack()

# Run the main loop
window.mainloop()
