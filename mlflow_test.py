import mlflow
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# ✅ 1. MLflow autolog 활성화
# 이 한 줄이 대부분의 작업을 자동으로 처리해줍니다.
mlflow.autolog(log_models=True)

# MLflow 서버 주소 및 실험 설정
mlflow.set_tracking_uri("http://localhost:18085")
mlflow.set_experiment("Pipeline with GridSearchCV")

# --- 데이터 준비 ---
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([1, 2.1, 2.9, 4.2, 5.1, 6, 7.3, 8.1])

# --- Scikit-learn 파이프라인 및 GridSearch 설정 ---

# 2. 전처리기와 모델을 파이프라인으로 묶기
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

# 3. 튜닝할 하이퍼파라미터 그리드 정의
# 파이프라인 단계 이름을 접두사로 사용 (예: 'regressor__alpha')
param_grid = {
    'regressor__alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
}

# 4. GridSearchCV 객체 생성
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')

# --- 실험 실행 ---
# autolog가 활성화되어 있으므로, start_run을 명시적으로 만들 필요도 없습니다.
print("Starting GridSearchCV with MLflow autologging...")

# 5. fit()을 호출하면 모든 것이 자동으로 기록됩니다.
grid_search.fit(X, y)

print("GridSearchCV finished.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best RMSE score: {-grid_search.best_score_}")