import sys 

sys.path.append("src")

from src.core.train_model import validate_data, get_data_from_files
import json

def test_validate_data():
    valid_record = {
        'temp_max': 30.5,
        'temp_afternoon': 25.3,
        'humidity_afternoon': 80
    }
    invalid_record_missing = {
        'temp_max': 30.5,
        'temp_afternoon': 25.3
    }
    invalid_record_type = {
        'temp_max': 'thirty',
        'temp_afternoon': 25.3,
        'humidity_afternoon': 80
    }

    assert validate_data(valid_record) == True
    assert validate_data(invalid_record_missing) == False
    assert validate_data(invalid_record_type) == False

def test_get_data_from_files(tmp_path):
    # Setup temporary JSON files
    valid_json = {"temp_max": 30, "temp_afternoon": 25, "humidity_afternoon": 80}
    invalid_json = {"temp_max": "thirty", "temp_afternoon": 25, "humidity_afternoon": 80}
    non_json_content = "Just some text"

    (tmp_path / "valid.json").write_text(json.dumps(valid_json))
    (tmp_path / "invalid.json").write_text(json.dumps(invalid_json))
    (tmp_path / "text.txt").write_text(non_json_content)

    data = get_data_from_files(str(tmp_path))
    assert len(data) == 1
    assert data[0] == valid_json
