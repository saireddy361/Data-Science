def mean_squared_error(y_true, y_pred):
 return np.mean((y_true - y_pred) ** 2)
def root_mean_squared_error(y_true, y_pred):
 return np.sqrt(mean_squared_error(y_true, y_pred))
def r2_score(y_true, y_pred):
 ss_res = np.sum((y_true - y_pred) ** 2)
 ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
 return 1 - (ss_res / ss_tot)
# Example Usage
mse = mean_squared_error(y, predictions)
rmse = root_mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)