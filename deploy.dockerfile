FROM python

RUN mkdir /app

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ['python', './frontend/app.py']


# TO RUN DOCKER
# docker build -f deploy.dockerfile -t mlcontainer:api .
# docker run -p 5000:5000 -d mlcontainer:api