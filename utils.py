import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None
