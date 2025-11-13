
import uvicorn
import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.main import router

load_dotenv(find_dotenv(), override=True)

app = FastAPI(
    debug=os.getenv('PROJECT_DEBUG'),
    title=os.getenv('PROJECT_NAME'),
    description=os.getenv('PROJECT_DESCRIPTION'),
    version=os.getenv('PROJECT_VERSION'),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def main():
    pass


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)
