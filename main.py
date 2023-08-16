import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = FastAPI()

# CORS Stuff
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    IQScore: float
    Height: float
    Weight: float
    Age: int
    CurEdu: int
    ExerciseDone: int
    Sex: int
    PP: int
    LF: int
    PH: int
    S: int
    BMIbin: int
    isHealthy: int


# NOTE: pickle error so train here
data = pd.read_csv('datasets/vinal_main.csv')
df = pd.DataFrame(data)
df = df.drop(['Unnamed: 0'], axis=1)
X = df.drop(['HADSbin'], axis=1)
y = df['HADSbin']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
model_decs = DecisionTreeClassifier(max_depth=15, random_state=0)
model_decs.fit(X_train, y_train)
# NOTE: train end


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.get("/status")
async def root():
    return {"status": "OK", "message": "Up & Running"}


@app.post("/predict")
async def predict(input: Data):
    dct = {k: [v] for k, v in dict(input).items()}
    input_df = pd.DataFrame(data=dct)
    answer = model_decs.predict(input_df)
    return {'status': "SUCCESS", 'result': answer[0]}
