from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, model_validator
from typing import List
import pandas as pd
import numpy as np
import re
import pickle
import io

app = FastAPI()

with open("/workspaces/ML_HSE/model_Ridge_df_prep_fe_cv_5_target_metric_neg_mean_squared_error.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("/workspaces/ML_HSE/ohe.pickle", "rb") as ohe_file:
    ohe = pickle.load(ohe_file)

with open("/workspaces/ML_HSE/scaler_fe.pickle", "rb") as scaler_fe_file:
    scaler_fe = pickle.load(scaler_fe_file)

class Item(BaseModel):
    name: str = 'Other'
    year: int = 2014
    selling_price: int | None = None
    km_driven: int = 70000
    fuel: str = 'Diesel'
    seller_type: str = 'Individual'
    transmission: str = 'Manual'
    owner: str = 'First Owner'
    mileage: str | float | None = 19.4
    engine: str | float = 1248.0
    max_power: str | float = 81.86
    torque_nm: float = 160.0
    max_torque_rpm: float = 2400.0
    seats: float = 5.0      
    
    @model_validator(mode='before')
    def pre_processing(cls, values) -> None:       
        values['torque_nm'], values['max_torque_rpm'] = cls.split_torque(values.pop('torque'))
        values['name'] = cls.update_name(values.pop('name'))
        values['owner'] = cls.merge_owner(values.pop('owner'))

        for k, v in cls.__pydantic_fields__.items():
            input_value = values.get(k, v.default)
            if pd.isna(input_value):
                values[k] = v.default
            elif k in ['mileage', 'engine', 'max_power']:
                values[k] = cls.convert_mileage_engine_max_power(input_value)
        return values

    def update_name(value):
        for k in name_map:
            if k.lower() in value.lower():
                return k
        else:
            return 'Other'
    
    def convert_mileage_engine_max_power(value):
        if isinstance(value, str):
            num_value = re.search('(\d+\.*\d*)', value)
            if num_value is None:
                return None
            else:
                num_value = num_value.group()
            dim_of_value = value.replace(num_value, '').strip()
            if dim_of_value.startswith('km') and dim_of_value.endswith('kg'):
                return float(num_value) * 1.4
            else:
                return float(num_value)
        elif isinstance(value, float | int):
            return float(value)
        return None
    
    def merge_owner(value):
        if value.strip() not in ['First Owner', 'Second Owner', 'Test Drive Car']:
            return 'Third & Above Owner'
        return value
    
    def split_torque(value):
        if pd.isna(value): return None, None
        res_nums = [float(i.replace(',', '')) for i in re.findall('([0-9]+[.,]*[0-9]*)\D*', value)]
        res_dims = re.findall('[a-zA-Z]+', value)
        assert 1 <= len(res_nums) <= 3
        if len(res_nums) == 2:
            torque_num, max_torque_rpm_num = res_nums[0], res_nums[-1]
        elif len(res_nums) >= 2:
            torque_num, max_torque_rpm_num = res_nums[0], sum(res_nums[-2:])/2
        else:
            torque_num, max_torque_rpm_num = res_nums[0], None
        if len(res_dims) >= 2:
            torque_dim, max_torque_rpm_dim = res_dims[0], res_dims[-1]
        else:
            return None, None
        assert max_torque_rpm_dim.lower() == 'rpm'
        if torque_dim.lower() == 'kgm':
            torque_num *= 9.80665
            torque_dim = 'Nm'
        return torque_num, max_torque_rpm_num


class Items(BaseModel):
    csv_data: List[Item]
    
    @model_validator(mode='before')
    def pre_processing(cls, values) -> None:
        values['csv_data'] = [Item(**i) for i in values.pop('csv_data')]
        return values


def np_cube(array):
    return np.pow(array, 3)

def np_outer(x):
    return np.outer(x, x).reshape(-1)

def np_outer_inv(x):
    return np.outer(x, np.reciprocal(x)).reshape(-1)

def np_matrix_outer(matrix):
    return np.apply_along_axis(np_outer, 1, matrix)

def np_matrix_outer_inv(matrix):
    return np.apply_along_axis(np_outer_inv, 1, matrix)

math_func = [
    np.log, 
    np_cube, 
    np.reciprocal, 
    np.sqrt, 
    np_matrix_outer, 
    np_matrix_outer_inv
    ]

def num_Feature_Engineering(values: np.array) -> np.array:
    values += + 1e-6
    features_gen = [func(values) for func in math_func]
    total_features = [values] + features_gen
    return np.concatenate(total_features, axis=1)

name_map = [
    'Renault', 'Maruti', 'Hyundai', 'Skoda', 'Ford', 'Fiat',
    'Mahindra', 'Tata', 'Honda', 'Toyota', 'Seat', 'Mercedes-Benz',
    'Jeep', 'Chevrolet', 'Datsun', 'Volkswagen', 'Nissan',
    'BMW', 'Audi', 'Volvo', 'Mitsubishi', 'DS'
    ]

num_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'torque_nm', 'max_torque_rpm']
cat_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']


@app.post("/predict_item")
def predict_item(item: Item):
    item_dict = item.model_dump()
    
    num_part = num_Feature_Engineering(
        np.array([item_dict[num_col] for num_col in num_columns]).reshape(1, -1),
    )
    cat_part = ohe.transform(
        [[item_dict[col_name] for col_name in cat_columns]]
        ).toarray()
    
    model_inputs = np.concatenate([scaler_fe.transform(num_part), cat_part.reshape(1, -1)], axis=1)
    return model.predict(model_inputs)[0]


@app.post("/predict_items")
async def predict_items(csv_file: UploadFile = File(...)):
    data = await csv_file.read()
    df = pd.read_csv(io.BytesIO(data))
    items = Items(csv_data=df.to_dict(orient='records')).model_dump()['csv_data']
    preds_data = pd.DataFrame(items)
    num_part = num_Feature_Engineering(preds_data[num_columns].values)
    cat_part = ohe.transform(preds_data[cat_columns]).toarray()
    model_inputs = np.concatenate([scaler_fe.transform(num_part), cat_part], axis=1)
    
    model_predicts = model.predict(model_inputs)
    df['predict'] = model_predicts

    return df.to_dict(orient='index')
