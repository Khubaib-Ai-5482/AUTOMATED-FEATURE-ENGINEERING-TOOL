import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn as nn

def preprocess_data(file_path, target_col, task_type):

    df = pd.read_csv(file_path)

    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(float)

    return X, y

def ml_feature_importance(X, y, task_type):

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        rf_importance = model.feature_importances_
        mi = mutual_info_classif(X, y)

    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        rf_importance = model.feature_importances_
        mi = np.zeros(X.shape[1])

    df = pd.DataFrame({
        "Feature": X.columns,
        "RF_Importance": rf_importance,
        "Mutual_Info": mi
    })

    df["Avg_score"] = (df["RF_Importance"] + df["Mutual_Info"]) / 2
    return df.sort_values("Avg_score", ascending=False)

class TabularNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.embedding = nn.Linear(64, 31)
        self.output = nn.Linear(31, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        embed = self.embedding(x)
        out = self.output(embed)
        return out, embed

def train_nn(X, y, task_type, epochs=50, lr=0.001):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y).to(device)

    if task_type == "classification":
        output_dim = len(np.unique(y))
        criterion = nn.CrossEntropyLoss()
    else:
        output_dim = 1
        criterion = nn.MSELoss()
        y_tensor = y_tensor.view(-1, 1)

    model = TabularNN(X.shape[1], output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds, _ = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print("epoch:", epoch + 1, "loss:", loss.item())

    with torch.no_grad():
        _, embeddings = model(X_tensor)

    return embeddings.cpu().numpy()


def run_tool():

    file_path = input("Enter CSV file path: ").strip()
    target_col = input("Enter target column: ").strip()
    task_type = input("Task type (classification / regression): ").strip().lower()

    print("preprocessing data")
    X, y = preprocess_data(file_path, target_col, task_type)

    print("feature importance")
    fi = ml_feature_importance(X, y, task_type)
    print(fi.head(10))

    print("training NN & making embeddings")
    embeddings = train_nn(X, y, task_type)
    
    embed_df = pd.DataFrame(
        embeddings,
        columns=[f"Embed_{i+1}" for i in range(embeddings.shape[1])]
    )
    final_df = pd.concat([X.reset_index(drop=True), embed_df], axis=1)
    final_df.to_csv("transformed_dataset.csv", index=False)

    print("file saved: transformed_dataset.csv")

if __name__ == "__main__":
    run_tool()
