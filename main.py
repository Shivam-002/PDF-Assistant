from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf_processor import process_pdf, upload_pdf
from GeminiLLM import query_llm

app = FastAPI()
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": f"{exc.detail}"},
    )


@app.post("/api/v1/pdf/upload")
def upload(file: UploadFile = File(...)) -> dict:
    """
    Endpoint to upload a PDF file.

    This endpoint allows users to upload a PDF file. If the file does not already exist,
    it will be saved to the server. If the file already exists, a message indicating so
    will be returned.

    Args:
    - file (UploadFile): The PDF file to be uploaded.

    Returns:
    - dict: A message indicating whether the upload was successful or if an error occurred.

    Raises:
    - Exception: If there is an issue with saving the file or any other exception occurs.
    """
    upload_pdf(file)

    return {"message": f"Successfully uploaded {file.filename}"}


@app.post("/api/v1/pdf/process")
def process(filename: str) -> dict:
    """
    Endpoint to process a PDF file.

    This endpoint allows users to process a PDF file by specifying the filename.
    The file processing logic should be implemented in the `process_pdf` function.

    Args:
    - filename (str): The name of the PDF file to be processed.

    Returns:
    - dict : A message indicating whether the processing was successful or if an error occurred.

    Raises:
    - Exception: If there is an issue with processing the file or any other exception occurs.
    """
    process_pdf(filename)
    return {"message": f"Successfully processed {filename}"}


@app.get("/api/v1/pdf/query")
def query(filename: str, query: str):
    print(f"Querying {filename} with {query}")

    results = query_llm(filename, query)
    return results
