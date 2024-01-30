import json
from sklearn.model_selection import train_test_split

#chia nho du lieu

def load_data_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def split_data(data, test_size=0.2, valid_size=0.05, random_state=None):
    # Chia thành tập train và tập test
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=[entry['target'] for entry in data])

    # Tính kích thước của tập validation
    valid_size_final = valid_size * len(data) / len(test_data)

    # Chia tập test thành tập valid và tập test
    valid_data, remaining_test_data = train_test_split(test_data, test_size=valid_size_final, random_state=random_state, stratify=[entry['target'] for entry in test_data])

    return train_data, valid_data, remaining_test_data

def save_data_to_json(data, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)

# Đọc dữ liệu từ file JSON
data = load_data_from_json("your_data.json")

# Chia dữ liệu thành tập train, valid, và test
train_data, valid_data, remaining_test_data = split_data(data, test_size=0.2, valid_size=0.05, random_state=42)

# Lưu các tập dữ liệu vào các file JSON
save_data_to_json(train_data, "train_data.json")
save_data_to_json(valid_data, "valid_data.json")
save_data_to_json(remaining_test_data, "test_data.json")
