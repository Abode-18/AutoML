    model_type = get_model_type(model)
    for column in num_columns:
        df_outliers = delete_outliers(df,column)
    if choosing_outlier_method(model,df,df_outliers) == None:
        df = df_outliers
        scaler = choose_scaler()
    else:
        scaler = choosing_outlier_method(model,df,df_outliers)
    info = get_info_about_column(df)
    creating_model(model,scaler,df)