import os
import logging

import jwt
from functools import wraps

from flask import Flask, request, jsonify
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone, timedelta


#from sklearn.datasets import load_iris
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split

#########configuração de segurança
JWT_SECRET = "MEUSEGREDOAQUI"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

#########configuração do log da API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo")

#########configuração de banco de dados
DB_URL = "sqlite:///predictions.db"

# Configuração para parametrizar a exibição dos comandos SQL
ECHO_SQL = os.getenv("ECHO_SQL", "False").lower() in ("true", "1", "t")
engine = create_engine(DB_URL, echo=False)

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

#Deinição do modelo de dados
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width  = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width  = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))


#Cria as tabelas no banco ( em produção utilizar Alembic)
Base.metadata.create_all(engine)

model = joblib.load("modelo_iris.pkl")
logger.info("Modelo carregado com sucesso.")


#### Iniciando a aplicação FLASK
app = Flask(__name__)
predictions_cache = {}

TEST_USERNAME = "admin"
TEST_PASSWORD = "secret"

def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def  token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # pegar token do header Authorization: Bearer <token>
        # decodificar e checar expiração
        return f(*args, **kwargs)
    return decorated

######## criação das rotas
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")
    if username == TEST_USERNAME and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({"token": token})
    else:
        return jsonify({"error":"Credenciais inválidas"}), 401
    
@app.route("/predict", methods=["POST"])
@token_required
def predict():
    """
    Endpoint protegido por token para obter predição.
    Corpo (JSON):
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """ 
    data = request.get_json(force=True)

    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    except (ValueError, KeyError) as e:
        logger.error("Dados de entrada inválidos: %s",e )
        return jsonify({"error":"Dados inválido, verifique parâmetros"}), 400
    
    #verifica se já esta no cache
    features = (sepal_length, sepal_width, petal_length, petal_width)
    if features in predictions_cache:
        logger.info("Cache hit para %s", features)
        predicted_class = predictions_cache[features]
    else:
        #Rodar Modelo
        input_data = np.array([features])
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        # Armazenar no cache
        predictions_cache[features] = predicted_class
        logger.info("Cache updated para %s", features)

    # Armazenar em DB
    db = SessionLocal()
    new_pred = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_class=predicted_class
    )   
    db.add(new_pred)
    db.commit()
    db.close()

    return jsonify({"prediction": predicted_class})

@app.route("/predictions", methods=["GET"])
@token_required
def list_predictions():
    """
    Lista as predições armazenadas no banco.
    Parâmetros opcionais ( via query string):
        - limit (int): quabtos registros retornar, padrão 10 
        - offset (int): a partir de qual registro começar, padrão 0 

    Exemplo:
        /predictions?limit=5&offset=10
      
    """
    limit = int(request.args.get("limit",10))
    offset = int(request.args.get("offset",0))
    db =SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset).all()
    db.close()

    results = []

    for p in preds:
        results.append({
            "id": p.id,
            "sepal_length": p.sepal_length,
            "sepal_width": p.sepal_width,
            "petal_length": p.petal_length,
            "petal_width": p.petal_width,
            "predicted_class": p.predicted_class,
            "created_at": p.created_at.isoformat()
        })
    
    return jsonify(results)


@app.route("/")
def route_hello():
    return jsonify({"message": "Hello, world!"})

if __name__ == "__main__":
    app.run(debug=True)





         




