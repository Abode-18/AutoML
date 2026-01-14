from sklearn.metrics import r2_score,accuracy_score



def evaluate(model,X_train,X_test,y_train,y_test):
    from core.modeling import get_model_type
    model.fit(X_train,y_train)
    if get_model_type(model) == "regression":
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test,y_pred)
    elif get_model_type(model) == "classification":
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test,y_pred)
    return score