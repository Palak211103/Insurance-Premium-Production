#Use Python 3.11 base image #its an operating system jiske andar python ka ek version run kar rha hota hai

FROM python:3.11-slim                    
#Set working directory
WORKDIR /app

#Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir  -r requirements.txt
#Copy rest of application code
COPY . .

#Expose the application port
EXPOSE 8000

#Command to start FastAPI application 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]