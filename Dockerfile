FROM python:3.9
# install python in the container

EXPOSE 8080 
# Expose the port 8000 in which our application runs

WORKDIR /app 
# Make /app as a working directory in the container

RUN pip install --no-cache-dir -U pip
# Install application dependencies from the requirements file

COPY ./requirements.txt .
# Copy requirements from host, to docker container in /app 

COPY . .
# Copy everything from ./project directory to /app in the container 

# Install the dependencies
RUN pip install -r requirements.txt

# execute the command python main.py (in the WORKDIR) to start the app
CMD ["python", "main.py"]






