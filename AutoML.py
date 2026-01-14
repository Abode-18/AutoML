from core.data_io import load_dataset
from core.data_info import dataset_info,column_statistics
from core.preprocessing import handling_outliers,choose_encoder
from core.modeling import creating_model,choose_model

def AutoML(path,y_column,model_name):

    df =load_dataset(path)
    info = dataset_info(df)
    cat_columns = info["cat_columns"]
    num_columns = info["num_columns"]
    stats = column_statistics(df,cat_columns,num_columns)
    model = choose_model(model_name,df,cat_columns,y_column)
    encoder = choose_encoder(df,model,cat_columns,y_column)
    df,scaler = handling_outliers(model,df,num_columns,cat_columns,stats,y_column,encoder)
    folder_path,score = creating_model(model,df,cat_columns,y_column,scaler,encoder,path)
    return folder_path,score