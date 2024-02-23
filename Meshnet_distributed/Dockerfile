FROM coinstacteam/coinstac-decentralized-test

# Set the working directory
WORKDIR /computation

# Copy the current directory contents into the container
COPY requirements.txt /computation

COPY ./NIB /computation/
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
# Copy the current directory contents into the container
COPY . /computation


RUN pip install --no-index --find-links=/computation/NIB nibabel

CMD ["python", "./scripts/entry.py"]


