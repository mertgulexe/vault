from fastapi import FastAPI


app = FastAPI()
@app.get("/")

def greet_json():
    return {"Hello": "World!"}


# this is a demo. Run: `./app/main.py` on port `8001` instead.