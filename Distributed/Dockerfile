FROM python:3.7.8

# Set the working directory
WORKDIR /computation

# Copy the current directory contents into the container
COPY requirements.txt /computation

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . /computation

CMD ["python", "entry.py"]
