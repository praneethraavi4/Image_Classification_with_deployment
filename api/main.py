from fastapi import FastAPI
import uvicorn
from PIL import Image
app = FastAPI()

@app.get("/ping")
async def ping():
    return "I am alive"

def get_image():
    Image.open()



@app.post("/predict")
async def predict():


    return "I am alive"




if __name__ == "__main__":
    uvicorn.run(app,host = "localhost",port = 8000)